//! Implement the derive(Orientation) macro

use proc_macro::TokenStream;
use quote::{quote, quote_spanned};
use syn::{Data, DeriveInput, Fields, Type};

/// Implement the derive(orientation) macro.
pub(crate) fn orientation(input: DeriveInput) -> TokenStream {
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    let name = &input.ident;

    let orientation_type = match input.data {
        Data::Struct(data) => {
            if let Ok(type_) = get_orientation_type(&data.fields) {
                type_
            } else {
                return quote_spanned! {
                    name.span() =>
                    compile_error!("derive(Orientation) requires a field named orientation.");
                }
                .into();
            }
        }
        Data::Enum(_) | Data::Union(_) => {
            return quote_spanned! {
                name.span() =>
                compile_error!("derive(Orientation) applies only to struct types.");
            }
            .into();
        }
    };

    let generated = quote! {
        impl #impl_generics ::hoomd_microstate::property::Orientation for #name #ty_generics #where_clause {
            type orientation = #orientation_type;

            fn orientation(&self) -> &Self::orientation {
                &self.orientation
            }

            fn orientation_mut(&mut self) -> &mut Self::orientation {
                &mut self.orientation
            }
        }
    };
    generated.into()
}

/// Get the type of the field named `orientation`.
fn get_orientation_type(fields: &Fields) -> Result<Type, ()> {
    for field in fields {
        if let Some(ref ident) = field.ident
            && ident == "orientation"
        {
            return Ok(field.ty.clone());
        }
    }
    Err(())
}
