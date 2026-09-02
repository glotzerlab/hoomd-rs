// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement the derive(DeltaEnergyOne) macro

use proc_macro2::{Span, TokenStream};
use quote::{quote, quote_spanned};
use syn::{Data, DeriveInput, Fields, GenericParam, Ident, Index, parse_quote, spanned::Spanned};

/// Implement the derive(DeltaEnergyOne) macro.
pub(crate) fn delta_energy_one(input: DeriveInput) -> TokenStream {
    let name = &input.ident;

    let data = match input.data {
        Data::Struct(data) => data,
        Data::Enum(_) | Data::Union(_) => {
            return quote_spanned! {
                name.span() =>
                compile_error!("derive(DeltaEnergyOne) applies only to struct types.");
            };
        }
    };

    let sum = delta_energy_one_sum(&data.fields);

    let mut generics = input.generics.clone();
    let b_ident = Ident::new("__B", Span::call_site());
    let s_ident = Ident::new("__S", Span::call_site());
    let x_ident = Ident::new("__X", Span::call_site());
    let c_ident = Ident::new("__C", Span::call_site());
    generics.params = [
        GenericParam::Type(b_ident.into()),
        GenericParam::Type(s_ident.into()),
        GenericParam::Type(x_ident.into()),
        GenericParam::Type(c_ident.into()),
    ]
    .into_iter()
    .chain(generics.params)
    .collect();

    // The user provided predicates may or may not end in a comma.
    // Therefore, list the additional generics first with a trailing
    // comma (the `,*,`) and then list the user provided predicates.
    let field_types = data.fields.iter().map(|f| f.ty.clone());
    if let Some(previous_where_clause) = generics.where_clause {
        let predicates = previous_where_clause.predicates;
        generics.where_clause = Some(parse_quote!(where
        #(#field_types: ::hoomd_interaction::DeltaEnergyOne<__B, __S, __X, __C>),*,
        #predicates
        ));
    } else {
        generics.where_clause = Some(parse_quote!(where
            #(#field_types: ::hoomd_interaction::DeltaEnergyOne<__B, __S, __X, __C>),*));
    }

    let (impl_generics, _, where_clause) = generics.split_for_impl();
    // Don't include the added generics when naming the struct type.
    let (_, ty_generics, _) = input.generics.split_for_impl();

    let generated = quote! {
        impl #impl_generics ::hoomd_interaction::DeltaEnergyOne<__B, __S, __X, __C> for #name #ty_generics #where_clause {
            #[inline]
            fn delta_energy_one(
                &self,
                initial_microstate: &::hoomd_microstate::Microstate<__B, __S, __X, __C>,
                body_index: usize,
                final_body: &::hoomd_microstate::Body<__B, __S>,
            ) -> f64 {
                #sum
            }
        }
    };
    generated
}

/// Sum the delta energy from all terms in the Hamiltonian.
fn delta_energy_one_sum(fields: &Fields) -> TokenStream {
    match fields {
        Fields::Named(fields) => {
            let terms = fields.named.iter().map(|f| {
                let name = &f.ident;
                quote_spanned! {f.span()=>
                    ::hoomd_interaction::DeltaEnergyOne::delta_energy_one(&self.#name,
                        initial_microstate, body_index, final_body)
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
                    ::hoomd_interaction::DeltaEnergyOne::delta_energy_one(&self.#index,
                        initial_microstate, body_index, final_body)
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
