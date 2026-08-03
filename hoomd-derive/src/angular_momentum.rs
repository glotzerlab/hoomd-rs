// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement the derive(AngularMomentum) macro

use proc_macro::TokenStream;
use quote::{quote, quote_spanned};
use syn::{Data, DeriveInput, Fields, Type};

/// Implement the derive(AngularMomentum) macro.
pub(crate) fn angular_momentum(input: DeriveInput) -> TokenStream {
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    let name = &input.ident;

    let angular_momentum_type = match input.data {
        Data::Struct(data) => {
            if let Ok(type_) = get_angular_momentum_type(&data.fields) {
                type_
            } else {
                return quote_spanned! {
                    name.span() =>
                    compile_error!("derive(AngularMomentum) requires a field named angular_momentum.");
                }
                .into();
            }
        }
        Data::Enum(_) | Data::Union(_) => {
            return quote_spanned! {
                name.span() =>
                compile_error!("derive(AngularMomentum) applies only to struct types.");
            }
            .into();
        }
    };

    let generated = quote! {
        impl #impl_generics ::hoomd_microstate::property::AngularMomentum for #name #ty_generics #where_clause {
            type AngularMomentum = #angular_momentum_type;

            #[inline]
            fn angular_momentum(&self) -> &Self::AngularMomentum {
                &self.angular_momentum
            }

            #[inline]
            fn angular_momentum_mut(&mut self) -> &mut Self::AngularMomentum {
                &mut self.angular_momentum
            }
        }
    };
    generated.into()
}

/// Get the type of the field named `angular_momentum`.
fn get_angular_momentum_type(fields: &Fields) -> Result<Type, ()> {
    for field in fields {
        if let Some(ref ident) = field.ident
            && ident == "angular_momentum"
        {
            return Ok(field.ty.clone());
        }
    }
    Err(())
}
