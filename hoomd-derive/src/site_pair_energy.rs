// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement the derive(DeltaEnergyOne) macro

use proc_macro2::{Span, TokenStream};
use quote::{quote, quote_spanned};
use syn::{Data, DeriveInput, Fields, GenericParam, Ident, Index, parse_quote, spanned::Spanned};

/// Implement the derive(DeltaEnergyOne) macro.
pub(crate) fn site_pair_energy(input: DeriveInput) -> TokenStream {
    let name = &input.ident;

    let data = match input.data {
        Data::Struct(data) => data,
        Data::Enum(_) | Data::Union(_) => {
            return quote_spanned! {
                name.span() =>
                compile_error!("derive(SitePairEnergy) applies only to struct types.");
            };
        }
    };
    
    let mut generics = input.generics.clone();
    let s_ident = Ident::new("__S", Span::call_site());
    generics.params = [
        GenericParam::Type(s_ident.into()),
    ]
    .into_iter()
    .chain(generics.params)
    .collect();

    let field_types = data.fields.iter().map(|f| f.ty.clone());
    if let Some(previous_where_clause) = generics.where_clause {
        let predicates = previous_where_clause.predicates;
        generics.where_clause = Some(parse_quote!(where
        #predicates,
        #(#field_types: ::hoomd_interaction::SitePairEnergy<__S>),*
        ));
    } else {
        generics.where_clause = Some(parse_quote!(where
            #(#field_types: ::hoomd_interaction::SitePairEnergy<__S>),*));
    }

    let (impl_generics, _, where_clause) = generics.split_for_impl();
    // Don't include the added generics when naming the struct type.
    let (_, ty_generics, _) = input.generics.split_for_impl();

    let site_pair_energy_sum = site_pair_energy_sum(&data.fields);
    let site_pair_energy_initial_sum = site_pair_energy_initial_sum(&data.fields);
    let is_only_infinite_or_zero = is_only_infinite_or_zero(&data.fields);

    let generated = quote! {
        impl #impl_generics ::hoomd_interaction::SitePairEnergy<__S> for #name #ty_generics #where_clause {
            #[inline]
            fn site_pair_energy(&self,
                site_properties_i: &__S,
                site_properties_j: &__S) -> f64 {
                #site_pair_energy_sum
            }

            #[inline]
            fn site_pair_energy_initial(&self,
                site_properties_i: &__S,
                site_properties_j: &__S) -> f64 {
                #site_pair_energy_initial_sum
            }

            #[inline]
            fn is_only_infinite_or_zero() -> bool {
                #is_only_infinite_or_zero
            }
            
        }
    };
    generated
}

/// Sum `site_pair_energy` over all fields.
fn site_pair_energy_sum(fields: &Fields) -> TokenStream {
    match fields {
        Fields::Named(fields) => {
            let terms = fields.named.iter().map(|f| {
                let name = &f.ident;
                quote_spanned! {f.span()=>
                    ::hoomd_interaction::SitePairEnergy::site_pair_energy(&self.#name,
                        site_properties_i, site_properties_j)
                }
            });

            quote! {
                let mut total = 0.0_f64;
                #(
                total += #terms;

                if total == f64::INFINITY {
                    return total;
                }
                )*
                total
            }
        }
        Fields::Unnamed(fields) => {
            let terms = fields.unnamed.iter().enumerate().map(|(i, f)| {
                let index = Index::from(i);
                quote_spanned! {f.span()=>
                    ::hoomd_interaction::SitePairEnergy::site_pair_energy(&self.#index,
                        site_properties_i, site_properties_j)
                }
            });

            quote! {
                let mut total = 0.0_f64;
                #(
                total += #terms;

                if total == f64::INFINITY {
                    return total;
                }
                )*
                total
            }
        }
        Fields::Unit => {
            quote!(0.0_f64)
        }
    }
}

/// Sum `site_pair_energy_initial` over all fields.
fn site_pair_energy_initial_sum(fields: &Fields) -> TokenStream {
    match fields {
        Fields::Named(fields) => {
            let terms = fields.named.iter().map(|f| {
                let name = &f.ident;
                quote_spanned! {f.span()=>
                    ::hoomd_interaction::SitePairEnergy::site_pair_energy_initial(&self.#name,
                        site_properties_i, site_properties_j)
                }
            });

            quote! {
                let mut total = 0.0_f64;
                #(
                total += #terms;

                if total == f64::INFINITY {
                    return total;
                }
                )*
                total
            }
        }
        Fields::Unnamed(fields) => {
            let terms = fields.unnamed.iter().enumerate().map(|(i, f)| {
                let index = Index::from(i);
                quote_spanned! {f.span()=>
                    ::hoomd_interaction::SitePairEnergy::site_pair_energy_initial(&self.#index,
                        site_properties_i, site_properties_j)
                }
            });

            quote! {
                let mut total = 0.0_f64;
                #(
                total += #terms;

                if total == f64::INFINITY {
                    return total;
                }
                )*
                total
            }
        }
        Fields::Unit => {
            quote!(0.0_f64)
        }
    }
}

/// And together `is_only_infinite_or_zero` over all fields.
fn is_only_infinite_or_zero(fields: &Fields) -> TokenStream {
    match fields {
        Fields::Named(fields) => {
            let terms = fields.named.iter().map(|f| {
                let ty = &f.ty;
                quote_spanned! {f.span()=>
                    #ty::is_only_infinite_or_zero()
                }
            });

            // quote! {
            //     let mut result = true;
            //     #(result = result && #terms;)*
            //     result
            // }
            quote! {
                #(#terms)&&*
            }
        }
        Fields::Unnamed(fields) => {
            let terms = fields.unnamed.iter().map(|f| {
                let ty = &f.ty;
                quote_spanned! {f.span()=>
                    #ty::is_only_infinite_or_zero()
                }
            });

            quote! {
                #(#terms)&&*
            }
        }
        Fields::Unit => {
            quote!(0.0_f64)
        }
    }
}
