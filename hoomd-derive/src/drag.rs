// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement the derive(Drag) macro

use proc_macro::TokenStream;
use quote::{quote, quote_spanned};
use syn::{Data, DeriveInput, Fields, Type};

/// Implement the derive(Drag) macro.
pub(crate) fn drag(input: DeriveInput) -> TokenStream {
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    let name = &input.ident;

    let _ = match input.data {
        Data::Struct(data) => {
            if let Ok(type_) = get_drag_type(&data.fields) {
                type_
            } else {
                return quote_spanned! {
                    name.span() =>
                    compile_error!("derive(Drag) requires a field named drag.");
                }
                .into();
            }
        }
        Data::Enum(_) | Data::Union(_) => {
            return quote_spanned! {
                name.span() =>
                compile_error!("derive(Drag) applies only to struct types.");
            }
            .into();
        }
    };

    let generated = quote! {
        impl #impl_generics ::hoomd_microstate::property::Drag for #name #ty_generics #where_clause {
            #[inline]
            fn drag(&self) -> &f64 {
                &self.drag
            }

            #[inline]
            fn drag_mut(&mut self) -> &mut f64 {
                &mut self.drag
            }
        }
    };
    generated.into()
}

/// Get the type of the field named `drag`.
fn get_drag_type(fields: &Fields) -> Result<Type, ()> {
    for field in fields {
        if let Some(ref ident) = field.ident
            && ident == "drag"
        {
            return Ok(field.ty.clone());
        }
    }
    Err(())
}
