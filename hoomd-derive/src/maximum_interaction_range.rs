//! Implement the derive(Position) macro

use proc_macro::TokenStream;
use quote::{quote, quote_spanned};
use syn::{Data, DeriveInput};

/// Implement the derive(MaximumInteractionRange) macro.
pub(crate) fn maximum_interaction_range(input: &DeriveInput) -> TokenStream {
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    let name = &input.ident;

    match input.data {
        Data::Struct(_) => {
            quote! {
                impl #impl_generics ::hoomd_interaction::MaximumInteractionRange for #name #ty_generics #where_clause {
                    fn maximum_interaction_range(&self) -> f64 {
                        self.maximum_interaction_range
                    }
                }
            }.into()
        },
        Data::Enum(_) | Data::Union(_) => {
            quote_spanned! {
                name.span() =>
                compile_error!("derive(Position) applies only to struct types.");
            }.into()
        },
        }
}
