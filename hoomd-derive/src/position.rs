// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement the derive(Position) macro

use proc_macro::TokenStream;
use quote::{quote, quote_spanned};
use syn::{Data, DeriveInput, Fields, Type};

/// Implement the derive(Position) macro.
pub(crate) fn position(input: DeriveInput) -> TokenStream {
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    let name = &input.ident;

    let position_type = match input.data {
        Data::Struct(data) => {
            if let Ok(type_) = get_position_type(&data.fields) {
                type_
            } else {
                return quote_spanned! {
                    name.span() =>
                    compile_error!("derive(Position) requires a field named position.");
                }
                .into();
            }
        }
        Data::Enum(_) | Data::Union(_) => {
            return quote_spanned! {
                name.span() =>
                compile_error!("derive(Position) applies only to struct types.");
            }
            .into();
        }
    };

    let generated = quote! {
        impl #impl_generics ::hoomd_microstate::property::Position for #name #ty_generics #where_clause {
            type Position = #position_type;

            #[inline]
            fn position(&self) -> &Self::Position {
                &self.position
            }

            #[inline]
            fn position_mut(&mut self) -> &mut Self::Position {
                &mut self.position
            }
        }
    };
    generated.into()
}

/// Get the type of the field named `position`.
fn get_position_type(fields: &Fields) -> Result<Type, ()> {
    for field in fields {
        if let Some(ref ident) = field.ident
            && ident == "position"
        {
            return Ok(field.ty.clone());
        }
    }
    Err(())
}
