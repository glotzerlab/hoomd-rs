// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement the derive(MaximumInteractionRange) macro

use proc_macro2::TokenStream;
use quote::{quote, quote_spanned};
use syn::{Data, DeriveInput, Fields, Index, parse_quote, spanned::Spanned};

/// Implement the derive(MaximumInteractionRange) macro.
pub(crate) fn maximum_interaction_range(input: DeriveInput) -> TokenStream {
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    let name = &input.ident;

    let data = match input.data {
        Data::Struct(data) => data,
        Data::Enum(_) | Data::Union(_) => {
            return quote_spanned! {
                name.span() =>
                compile_error!("derive(MaximumInteractionRange) applies only to struct types.");
            }.into()
        },
    };

    if has_maximum_interaction_range(&data.fields) {
        quote! {
            impl #impl_generics ::hoomd_interaction::MaximumInteractionRange for #name #ty_generics #where_clause {
                #[inline]
                fn maximum_interaction_range(&self) -> f64 {
                    self.maximum_interaction_range
                }
            }
        }
    } else {
        let max = maximum_interaction_range_max(&data.fields);

        let mut generics = input.generics.clone();
        let field_types = data.fields.iter().map(|f| f.ty.clone());
        if let Some(previous_where_clause) = generics.where_clause {
            let predicates = previous_where_clause.predicates;
            generics.where_clause = Some(parse_quote!(where
                #predicates,
                #(#field_types: ::hoomd_interaction::MaximumInteractionRange),*
                ));
        } else {
            generics.where_clause = Some(parse_quote!(where
                #(#field_types: ::hoomd_interaction::MaximumInteractionRange),*));
        }
        let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

        quote! {
            impl #impl_generics ::hoomd_interaction::MaximumInteractionRange for #name #ty_generics #where_clause {
                #[inline]
                fn maximum_interaction_range(&self) -> f64 {
                    #max
                }
            }
        }
    }
}

/// Determine if there exists a field named `maximum_interaction_range`.
fn has_maximum_interaction_range(fields: &Fields) -> bool {
    for field in fields {
        if let Some(ref ident) = field.ident
            && ident == "maximum_interaction_range"
        {
            return true;
        }
    }
    false
}

/// Get the maximum interaction range from all terms in the Hamiltonian.
fn maximum_interaction_range_max(fields: &Fields) -> TokenStream {
    match fields {
        Fields::Named(fields) => {
            let terms = fields.named.iter().map(|f| {
                let name = &f.ident;
                quote_spanned! {f.span()=>
                    ::hoomd_interaction::MaximumInteractionRange::maximum_interaction_range(&self.#name)
                }
            });
            
            quote! {
                let mut total = 0.0_f64;
                #(total = total.max(#terms);)*
                total
            }
        }
        Fields::Unnamed(fields) => {
            let terms = fields.unnamed.iter().enumerate().map(|(i, f)| {
                let index = Index::from(i);
                quote_spanned! {f.span()=>
                    ::hoomd_interaction::MaximumInteractionRange::maximum_interaction_range(&self.#index)
                }
            });

            quote! {
                let mut total = 0.0_f64;
                #(total = total.max(#terms);)*
                total
            }
        }
        Fields::Unit => {
            quote!(0.0_f64)
        }
    }
}
