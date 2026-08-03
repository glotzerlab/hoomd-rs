// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement the derive(MomentOfInertia) macro

use proc_macro::TokenStream;
use quote::{quote, quote_spanned};
use syn::{Data, DeriveInput, Fields, Type};

/// Implement the derive(MomentOfInertia) macro.
pub(crate) fn moment_of_inertia(input: DeriveInput) -> TokenStream {
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    let name = &input.ident;

    let moment_of_inertia_type = match input.data {
        Data::Struct(data) => {
            if let Ok(type_) = get_moment_of_inertia_type(&data.fields) {
                type_
            } else {
                return quote_spanned! {
                    name.span() =>
                    compile_error!("derive(MomentOfInertia) requires a field named moment_of_inertia.");
                }
                .into();
            }
        }
        Data::Enum(_) | Data::Union(_) => {
            return quote_spanned! {
                name.span() =>
                compile_error!("derive(MomentOfInertia) applies only to struct types.");
            }
            .into();
        }
    };

    let generated = quote! {
        impl #impl_generics ::hoomd_microstate::property::MomentOfInertia for #name #ty_generics #where_clause {
            type MomentOfInertia = #moment_of_inertia_type;

            #[inline]
            fn moment_of_inertia(&self) -> &Self::MomentOfInertia {
                &self.moment_of_inertia
            }

            #[inline]
            fn moment_of_inertia_mut(&mut self) -> &mut Self::MomentOfInertia {
                &mut self.moment_of_inertia
            }
        }
    };
    generated.into()
}

/// Get the type of the field named `moment_of_inertia`.
fn get_moment_of_inertia_type(fields: &Fields) -> Result<Type, ()> {
    for field in fields {
        if let Some(ref ident) = field.ident
            && ident == "moment_of_inertia"
        {
            return Ok(field.ty.clone());
        }
    }
    Err(())
}
