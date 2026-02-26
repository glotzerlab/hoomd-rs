// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement the derive(DeltaEnergyInsert) macro

use proc_macro2::{Span, TokenStream};
use quote::{quote, quote_spanned};
use syn::{Data, DeriveInput, Fields, GenericParam, Ident, Index, parse_quote, spanned::Spanned};

/// Implement the derive(DeltaEnergyInsert) macro.
pub(crate) fn delta_energy_insert(input: DeriveInput) -> TokenStream {
    let name = &input.ident;

    let data = match input.data {
        Data::Struct(data) => data,
        Data::Enum(_) | Data::Union(_) => {
            return quote_spanned! {
                name.span() =>
                compile_error!("derive(DeltaEnergyInsert) applies only to struct types.");
            };
        }
    };

    let sum = delta_energy_insert_sum(&data.fields);

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

    let field_types = data.fields.iter().map(|f| f.ty.clone());
    if let Some(previous_where_clause) = generics.where_clause {
        let predicates = previous_where_clause.predicates;
        generics.where_clause = Some(parse_quote!(where
        #predicates,
        #(#field_types: ::hoomd_interaction::DeltaEnergyInsert<__B, __S, __X, __C>),*
        ));
    } else {
        generics.where_clause = Some(parse_quote!(where
            #(#field_types: ::hoomd_interaction::DeltaEnergyInsert<__B, __S, __X, __C>),*));
    }

    let (impl_generics, _, where_clause) = generics.split_for_impl();
    // Don't include the added generics when naming the struct type.
    let (_, ty_generics, _) = input.generics.split_for_impl();

    let generated = quote! {
        impl #impl_generics ::hoomd_interaction::DeltaEnergyInsert<__B, __S, __X, __C> for #name #ty_generics #where_clause {
            #[inline]
            fn delta_energy_insert(
                &self,
                initial_microstate: &::hoomd_microstate::Microstate<__B, __S, __X, __C>,
                new_body: &::hoomd_microstate::Body<__B, __S>,
            ) -> f64 {
                #sum
            }
        }
    };
    generated
}

/// Sum the delta energy from all terms in the Hamiltonian.
fn delta_energy_insert_sum(fields: &Fields) -> TokenStream {
    match fields {
        Fields::Named(fields) => {
            let terms = fields.named.iter().map(|f| {
                let name = &f.ident;
                quote_spanned! {f.span()=>
                    ::hoomd_interaction::DeltaEnergyInsert::delta_energy_insert(&self.#name,
                        initial_microstate, new_body)
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
                    ::hoomd_interaction::DeltaEnergyInsert::delta_energy_insert(&self.#index,
                        initial_microstate, new_body)
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
