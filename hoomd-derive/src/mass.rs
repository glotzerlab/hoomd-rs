// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement the derive(Mass) macro

use proc_macro::TokenStream;
use quote::{quote, quote_spanned};
use syn::{Data, DeriveInput, Fields, Type};

/// Implement the derive(Mass) macro.
pub(crate) fn mass(input: DeriveInput) -> TokenStream {
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    let name = &input.ident;

    let _ = match input.data {
        Data::Struct(data) => {
            if let Ok(type_) = get_mass_type(&data.fields) {
                type_
            } else {
                return quote_spanned! {
                    name.span() =>
                    compile_error!("derive(Mass) requires a field named mass.");
                }
                .into();
            }
        }
        Data::Enum(_) | Data::Union(_) => {
            return quote_spanned! {
                name.span() =>
                compile_error!("derive(Mass) applies only to struct types.");
            }
            .into();
        }
    };

    let generated = quote! {
        impl #impl_generics ::hoomd_microstate::property::Mass for #name #ty_generics #where_clause {
            #[inline]
            fn mass(&self) -> f64 {
                self.mass
            }
        }
    };
    generated.into()
}

/// Get the type of the field named `mass`.
fn get_mass_type(fields: &Fields) -> Result<Type, ()> {
    for field in fields {
        if let Some(ref ident) = field.ident
            && ident == "mass"
        {
            return Ok(field.ty.clone());
        }
    }
    Err(())
}
