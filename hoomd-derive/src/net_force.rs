// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement the derive(NetForce) macro

use proc_macro::TokenStream;
use quote::{quote, quote_spanned};
use syn::{Data, DeriveInput, Fields, Type};

/// Implement the derive(NetForce) macro.
pub(crate) fn net_force(input: DeriveInput) -> TokenStream {
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    let name = &input.ident;

    let net_force_type = match input.data {
        Data::Struct(data) => {
            if let Ok(type_) = get_net_force_type(&data.fields) {
                type_
            } else {
                return quote_spanned! {
                    name.span() =>
                    compile_error!("derive(NetForce) requires a field named net_force.");
                }
                .into();
            }
        }
        Data::Enum(_) | Data::Union(_) => {
            return quote_spanned! {
                name.span() =>
                compile_error!("derive(NetForce) applies only to struct types.");
            }
            .into();
        }
    };

    let generated = quote! {
        impl #impl_generics ::hoomd_microstate::property::NetForce for #name #ty_generics #where_clause {
            type NetForce = #net_force_type;

            #[inline]
            fn net_force(&self) -> &Self::NetForce {
                &self.net_force
            }

            #[inline]
            fn net_force_mut(&mut self) -> &mut Self::NetForce {
                &mut self.net_force
            }
        }
    };
    generated.into()
}

/// Get the type of the field named `net_force`.
fn get_net_force_type(fields: &Fields) -> Result<Type, ()> {
    for field in fields {
        if let Some(ref ident) = field.ident
            && ident == "net_force"
        {
            return Ok(field.ty.clone());
        }
    }
    Err(())
}
