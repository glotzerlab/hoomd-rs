// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement the derive(TotalEnergy) macro


use proc_macro2::{Span, TokenStream};
use quote::{quote, quote_spanned};
use syn::{Data, DeriveInput, Fields, GenericParam, Ident, Index, parse_quote, spanned::Spanned};

/// Implement the derive(TotalEnergy) macro.
pub(crate) fn total_energy(input: DeriveInput) -> TokenStream {
    let name = &input.ident;

    let data = match input.data {
        Data::Struct(data) => data,
        Data::Enum(_) | Data::Union(_) => {
            return quote_spanned! {
                name.span() =>
                compile_error!("derive(TotalEnergy) applies only to struct types.");
            }
        }
    };

    let sum = total_energy_sum(&data.fields);

    let mut generics = input.generics.clone();
    let m_ident = Ident::new("__M", Span::call_site());
    generics.params = [GenericParam::Type(m_ident.into())]
                .into_iter()
                .chain(generics.params)
                .collect();

    let field_types = data.fields.iter().map(|f| f.ty.clone());
    if let Some(previous_where_clause) = generics.where_clause {
        let predicates = previous_where_clause.predicates;
        generics.where_clause = Some(parse_quote!(where
            #predicates,
            #(#field_types: ::hoomd_interaction::TotalEnergy<__M>),*
            ));
    } else {
        generics.where_clause = Some(parse_quote!(where
            #(#field_types: ::hoomd_interaction::TotalEnergy<__M>),*));
    }

    let (impl_generics, _, where_clause) = generics.split_for_impl();
    // Don't include the added generics when naming the struct type.
    let (_, ty_generics, _) = input.generics.split_for_impl();
    
    let generated = quote! {
        impl #impl_generics ::hoomd_interaction::TotalEnergy<__M> for #name #ty_generics #where_clause {
            #[inline]
            fn total_energy(&self, microstate: &__M) -> f64 {
                #sum
            }
        }
    };
    generated
}

/// Sum the total energy from all terms in the Hamiltonian.
fn total_energy_sum(fields: &Fields) -> TokenStream {
    match fields {
        Fields::Named(fields) => {
            let terms = fields.named.iter().map(|f| {
                let name = &f.ident;
                quote_spanned! {f.span()=>
                    ::hoomd_interaction::TotalEnergy::total_energy(&self.#name, microstate)
                }
            });
            
            quote! {
                let mut total = 0.0_f64;
                #(
                total = total + #terms;

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
                    ::hoomd_interaction::TotalEnergy::total_energy(&self.#index, microstate)
                }
            });

            quote! {
                let mut total = 0.0_f64;
                #(
                total = total + #terms;

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
