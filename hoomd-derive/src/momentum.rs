// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Implement the derive(Momentum) macro

use proc_macro::TokenStream;
use quote::{quote, quote_spanned};
use syn::{Data, DeriveInput, Fields, Type, WhereClause};

/// Implement the derive(Momentum) macro.
pub(crate) fn momentum(input: DeriveInput) -> TokenStream {
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    let name = &input.ident;

    let momentum_type = match input.data {
        Data::Struct(data) => {
            if let Ok(type_) = get_momentum_type(&data.fields) {
                type_
            } else {
                return quote_spanned! {
                    name.span() =>
                    compile_error!("derive(Momentum) requires a field named momentum.");
                }
                .into();
            }
        }
        Data::Enum(_) | Data::Union(_) => {
            return quote_spanned! {
                name.span() =>
                compile_error!("derive(Momentum) applies only to struct types.");
            }
            .into();
        }
    };

    let mut other_where: WhereClause = syn::parse_quote! { 
        where
            #momentum_type: std::ops::Mul<f64, Output = #momentum_type>
            + std::ops::Div<f64, Output = #momentum_type>
            + Copy
    };

    let final_where_clause = match where_clause {
        Some(original) => {
            other_where.predicates.extend(original.predicates.clone());
            other_where
        }
        None => other_where
    };

    let generated = quote! {
        impl #impl_generics ::hoomd_microstate::property::Momentum for #name #ty_generics #final_where_clause {
            type Momentum = #momentum_type;

            #[inline]
            fn momentum(&self) -> &Self::Momentum {
                &self.momentum
            }

            #[inline]
            fn momentum_mut(&mut self) -> &mut Self::Momentum {
                &mut self.momentum
            }

            #[inline]
            fn velocity(&self) -> Self::Momentum {
                self.momentum / <Self as hoomd_microstate::property::Mass>::mass(&self)
            }

            #[inline]
            fn set_velocity(&mut self, velocity: Self::Momentum) {
                *self.momentum_mut() = velocity * <Self as hoomd_microstate::property::Mass>::mass(&self);
            }
        }
    };
    generated.into()
}

/// Get the type of the field named `momentum`.
fn get_momentum_type(fields: &Fields) -> Result<Type, ()> {
    for field in fields {
        if let Some(ref ident) = field.ident
            && ident == "momentum"
        {
            return Ok(field.ty.clone());
        }
    }
    Err(())
}
