// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement the derive(NetVirial) macro

use proc_macro::TokenStream;
use quote::{quote, quote_spanned};
use syn::{Data, DeriveInput, Fields, Type};

/// Implement the derive(NetVirial) macro.
pub(crate) fn net_virial(input: DeriveInput) -> TokenStream {
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    let name = &input.ident;

    let net_virial_type = match input.data {
        Data::Struct(data) => {
            if let Ok(type_) = get_net_virial_type(&data.fields) {
                type_
            } else {
                return quote_spanned! {
                    name.span() =>
                    compile_error!("derive(NetVirial) requires a field named net_virial.");
                }
                .into();
            }
        }
        Data::Enum(_) | Data::Union(_) => {
            return quote_spanned! {
                name.span() =>
                compile_error!("derive(NetVirial) applies only to struct types.");
            }
            .into();
        }
    };

    let generated = quote! {
        impl #impl_generics ::hoomd_microstate::property::NetVirial for #name #ty_generics #where_clause {
            type NetVirial = #net_virial_type;

            #[inline]
            fn net_virial(&self) -> &Self::NetVirial {
                &self.net_virial
            }

            #[inline]
            fn net_virial_mut(&mut self) -> &mut Self::NetVirial {
                &mut self.net_virial
            }
        }
    };
    generated.into()
}

/// Get the type of the field named `net_virial`.
fn get_net_virial_type(fields: &Fields) -> Result<Type, ()> {
    for field in fields {
        if let Some(ref ident) = field.ident
            && ident == "net_virial"
        {
            return Ok(field.ty.clone());
        }
    }
    Err(())
}
