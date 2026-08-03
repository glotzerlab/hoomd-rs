// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement the derive(RotationalDrag) macro

use proc_macro::TokenStream;
use quote::{quote, quote_spanned};
use syn::{Data, DeriveInput, Fields, Type};

/// Implement the derive(RotationalDrag) macro.
pub(crate) fn rotational_drag(input: DeriveInput) -> TokenStream {
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    let name = &input.ident;

    let rotational_drag_type = match input.data {
        Data::Struct(data) => {
            if let Ok(type_) = get_rotational_drag_type(&data.fields) {
                type_
            } else {
                return quote_spanned! {
                    name.span() =>
                    compile_error!("derive(RotationalDrag) requires a field named rotational_drag.");
                }
                .into();
            }
        }
        Data::Enum(_) | Data::Union(_) => {
            return quote_spanned! {
                name.span() =>
                compile_error!("derive(RotationalDrag) applies only to struct types.");
            }
            .into();
        }
    };

    let generated = quote! {
        impl #impl_generics ::hoomd_microstate::property::RotationalDrag for #name #ty_generics #where_clause {
            type RotationalDrag = #rotational_drag_type;

            #[inline]
            fn rotational_drag(&self) -> &Self::RotationalDrag {
                &self.rotational_drag
            }

            #[inline]
            fn rotational_drag_mut(&mut self) -> &mut Self::RotationalDrag {
                &mut self.rotational_drag
            }
        }
    };
    generated.into()
}

/// Get the type of the field named `rotational_drag`.
fn get_rotational_drag_type(fields: &Fields) -> Result<Type, ()> {
    for field in fields {
        if let Some(ref ident) = field.ident
            && ident == "rotational_drag"
        {
            return Ok(field.ty.clone());
        }
    }
    Err(())
}
