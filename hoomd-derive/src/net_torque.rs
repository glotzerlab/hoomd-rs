// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement the derive(NetTorque) macro

use proc_macro::TokenStream;
use quote::{quote, quote_spanned};
use syn::{Data, DeriveInput, Fields, Type};

/// Implement the derive(NetTorque) macro.
pub(crate) fn net_torque(input: DeriveInput) -> TokenStream {
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    let name = &input.ident;

    let net_torque_type = match input.data {
        Data::Struct(data) => {
            if let Ok(type_) = get_net_torque_type(&data.fields) {
                type_
            } else {
                return quote_spanned! {
                    name.span() =>
                    compile_error!("derive(NetTorque) requires a field named net_torque.");
                }
                .into();
            }
        }
        Data::Enum(_) | Data::Union(_) => {
            return quote_spanned! {
                name.span() =>
                compile_error!("derive(NetTorque) applies only to struct types.");
            }
            .into();
        }
    };

    let generated = quote! {
        impl #impl_generics ::hoomd_microstate::property::NetTorque for #name #ty_generics #where_clause {
            type NetTorque = #net_torque_type;

            #[inline]
            fn net_torque(&self) -> &Self::NetTorque {
                &self.net_torque
            }

            #[inline]
            fn net_torque_mut(&mut self) -> &mut Self::NetTorque {
                &mut self.net_torque
            }
        }
    };
    generated.into()
}

/// Get the type of the field named `net_torque`.
fn get_net_torque_type(fields: &Fields) -> Result<Type, ()> {
    for field in fields {
        if let Some(ref ident) = field.ident
            && ident == "net_torque"
        {
            return Ok(field.ty.clone());
        }
    }
    Err(())
}
