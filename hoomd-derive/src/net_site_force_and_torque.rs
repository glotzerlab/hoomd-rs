// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement the derive(NetSiteForce) macro

use proc_macro2::{Span, TokenStream};
use quote::{quote, quote_spanned};
use syn::{Data, DeriveInput, Fields, GenericParam, Ident, Index, parse_quote, spanned::Spanned};

/// Implement the derive(NetSiteForce) macro.
pub(crate) fn net_site_force_and_torque(input: DeriveInput) -> TokenStream {
    let name = &input.ident;

    let data = match input.data {
        Data::Struct(data) => data,
        Data::Enum(_) | Data::Union(_) => {
            return quote_spanned! {
                name.span() =>
                compile_error!("derive(NetSiteForceAndTorque) applies only to struct types.");
            };
        }
    };

    let sum = net_site_force_and_torque_sum(&data.fields);

    let mut generics = input.generics.clone();
    let v_ident = Ident::new("__V", Span::call_site());
    let b_ident = Ident::new("__B", Span::call_site());
    let s_ident = Ident::new("__S", Span::call_site());
    let x_ident = Ident::new("__X", Span::call_site());
    let c_ident = Ident::new("__C", Span::call_site());
    generics.params = [
        GenericParam::Type(v_ident.into()),
        GenericParam::Type(b_ident.into()),
        GenericParam::Type(s_ident.into()),
        GenericParam::Type(x_ident.into()),
        GenericParam::Type(c_ident.into()),
    ]
    .into_iter()
    .chain(generics.params)
    .collect();

    // The user provided predicates may or may not end in a comma.
    // Therefore, list the additional generics first with a trailing
    // comma (the `,*,`) and then list the user provided predicates.
    let field_types = data.fields.iter().map(|f| f.ty.clone());
    if let Some(previous_where_clause) = generics.where_clause {
        let predicates = previous_where_clause.predicates;
        generics.where_clause = Some(parse_quote!(where
        __V: ::std::ops::AddAssign<__V> + ::std::default::Default + ::hoomd_vector::Wedge,
        __V::Bivector: ::std::ops::AddAssign<__V::Bivector> + ::std::default::Default,
        __S: ::hoomd_microstate::property::Position<Position = __V>,
        #(#field_types: ::hoomd_interaction::NetSiteForceAndTorque<__B, __S, __X, __C, Force = __V>),*,
        #predicates
        ));
    } else {
        generics.where_clause = Some(parse_quote!(where
            __V: ::std::ops::AddAssign<__V> + ::std::default::Default + ::hoomd_vector::Wedge,
            __V::Bivector: ::std::ops::AddAssign<__V::Bivector> + ::std::default::Default,
            __S: ::hoomd_microstate::property::Position<Position = __V>,
            #(#field_types: ::hoomd_interaction::NetSiteForceAndTorque<__B, __S, __X, __C, Force = __V>),*
            ));
    }

    let (impl_generics, _, where_clause) = generics.split_for_impl();
    // Don't include the added generics when naming the struct type.
    let (_, ty_generics, _) = input.generics.split_for_impl();

    let generated = quote! {
        impl #impl_generics ::hoomd_interaction::NetSiteForceAndTorque<__B, __S, __X, __C> for #name #ty_generics #where_clause
            {
            type Force = __V;
            
            #[inline]
            fn net_site_force_and_torque(
                &self,
                microstate: &::hoomd_microstate::Microstate<__B, __S, __X, __C>,
                site_index: usize,
            ) -> (__V, __V::Bivector) {
                #sum
            }
        }
    };
    generated
}

/// Sum the net force over all terms in the interaction model.
fn net_site_force_and_torque_sum(fields: &Fields) -> TokenStream {
    match fields {
        Fields::Named(fields) => {
            let terms = fields.named.iter().map(|f| {
                let name = &f.ident;
                quote_spanned! {f.span()=>
                    ::hoomd_interaction::NetSiteForceAndTorque::net_site_force_and_torque(&self.#name,
                        microstate, site_index)
                }
            });

            quote! {
                let mut total_force = __V::default();
                let mut total_torque = __V::Bivector::default();

                #(
                let (force, torque) = #terms;

                total_force += force;
                total_torque += torque;
                )*
                (total_force, total_torque)                
            }
        }
        Fields::Unnamed(fields) => {
            let terms = fields.unnamed.iter().enumerate().map(|(i, f)| {
                let index = Index::from(i);
                quote_spanned! {f.span()=>
                    ::hoomd_interaction::NetSiteForceAndTorque::net_site_force_and_torque(&self.#index,
                        microstate, site_index)
                }
            });

            quote! {
                let mut total_force = __V::default();
                let mut total_torque = __V::Bivector::default();

                #(
                let (force, torque) = #terms;

                total_force += force;
                total_torque += torque;
                )*
                (total_force, total_torque)                
            }
        }
        Fields::Unit => {
            quote!((__V::default(), __V::Bivector::default()))
        }
    }
}
