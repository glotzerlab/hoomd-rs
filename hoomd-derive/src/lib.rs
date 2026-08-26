// Copyright (c) 2024-2026 The Regents of the University of Michigan.
// Part of hoomd-rs, released under the BSD 3-Clause License.

//! Derive macros for traits from a variety of hoomd-rs crates.
//!
//! # Complete documentation
//!
//! `hoomd-derive` is is a part of *hoomd-rs*. Read the [complete documentation]
//! for more information.
//!
//! [complete documentation]: https://hoomd-rs.readthedocs.io

#![allow(
    clippy::missing_inline_in_public_items,
    reason = "No need to inline macros"
)]

use proc_macro::TokenStream;
use syn::{
    DeriveInput,
    Fields,
    ItemStruct,
    Token,
    Type,
    WhereClause,
    parse::{Parse, ParseStream},
    parse_macro_input,
    parse_quote
};
use quote::quote;

mod angular_momentum;
mod delta_energy_insert;
mod delta_energy_one;
mod delta_energy_remove;
mod drag;
mod mass;
mod maximum_interaction_range;
mod moment_of_inertia;
mod momentum;
mod net_force;
mod net_site_force_and_virial;
mod net_site_force_virial_and_torque;
mod net_torque;
mod net_virial;
mod orientation;
mod position;
mod rotational_drag;
mod site_pair_energy;
mod total_energy;

/// Automatically implement the `hoomd_microstate::property::AngularMomentum` trait.
///
/// The derived implementation returns a reference to the structure's `angular_momentum`
/// field.
///
/// Valid on structs with named fields.
#[proc_macro_derive(AngularMomentum)]
pub fn angular_momentum_derive(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    angular_momentum::angular_momentum(input)
}

/// Automatically implement the `hoomd_interaction::DeltaEnergyInsert` trait.
///
/// The implemented `delta_energy_insert` sums the result of `delta_energy_insert`
/// over all fields. The implementation returns early when any one field returns
/// infinity.
///
/// Valid on:
/// * Structs with named fields.
/// * Tuple structs.
#[proc_macro_derive(DeltaEnergyInsert)]
pub fn delta_energy_insert_derive(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    delta_energy_insert::delta_energy_insert(input).into()
}

/// Automatically implement the `hoomd_interaction::DeltaEnergyOne` trait.
///
/// The implemented `delta_energy_one` sums the result of `delta_energy_one`
/// over all fields. The implementation returns early when any one field returns
/// infinity.
///
/// Valid on:
/// * Structs with named fields.
/// * Tuple structs.
#[proc_macro_derive(DeltaEnergyOne)]
pub fn delta_energy_one_derive(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    delta_energy_one::delta_energy_one(input).into()
}

/// Automatically implement the `hoomd_interaction::DeltaEnergyRemove` trait.
///
/// The implemented `delta_energy_remove` sums the result of `delta_energy_remove`
/// over all fields. The implementation returns early when any one field returns
/// infinity.
///
/// Valid on:
/// * Structs with named fields.
/// * Tuple structs.
#[proc_macro_derive(DeltaEnergyRemove)]
pub fn delta_energy_remove_derive(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    delta_energy_remove::delta_energy_remove(input).into()
}

/// Automatically implement the `hoomd_microstate::property::Drag` trait.
///
/// The derived implementation returns a reference to the structure's `drag`
/// field.
///
/// Valid on structs with named fields.
#[proc_macro_derive(Drag)]
pub fn drag_derive(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    drag::drag(input)
}

/// Automatically implement the `hoomd_microstate::property::Mass` trait.
///
/// The derived implementation returns a reference to the structure's `mass`
/// field.
///
/// Valid on structs with named fields.
#[proc_macro_derive(Mass)]
pub fn mass_derive(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    mass::mass(input)
}

/// Automatically implement the `hoomd_interaction::MaximumInteractionRange` trait.
///
/// If the type has a `maximum_interaction_range` field, the derived implementation
/// returns it. If the type does not, the derived implementation returns the
/// maximum of `maximum_interaction_range()` of each field.
///
/// Valid on:
/// * Structs with named fields.
/// * Tuple structs.
#[proc_macro_derive(MaximumInteractionRange)]
pub fn maximum_interaction_range_derive(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    maximum_interaction_range::maximum_interaction_range(input).into()
}

/// Automatically implement the `hoomd_microstate::property::MomentOfInertia` trait.
///
/// The derived implementation returns a reference to the structure's
/// `moment_of_inertia` field.
///
/// Valid on structs with named fields.
#[proc_macro_derive(MomentOfInertia)]
pub fn moment_of_inertia_derive(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    moment_of_inertia::moment_of_inertia(input)
}

/// Automatically implement the `hoomd_microstate::property::Momentum` trait.
///
/// The derived implementation returns a reference to the structure's `momentum`
/// field.
///
/// Valid on structs with named fields.
#[proc_macro_derive(Momentum)]
pub fn momentum_derive(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    momentum::momentum(input)
}

/// Automatically implement the `hoomd_microstate::property::NetForce` trait.
///
/// The derived implementation returns a reference to the structure's `net_force`
/// field.
///
/// Valid on structs with named fields.
#[proc_macro_derive(NetForce)]
pub fn net_force_derive(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    net_force::net_force(input)
}

/// Automatically implement the `hoomd_interaction::NetSiteForceAndVirial` trait.
///
/// The implemented `net_site_force_and_virial` sums the result of `net_site_force_and_virial`
/// over all fields.
///
/// Valid on:
/// * Structs with named fields.
/// * Tuple structs.
#[proc_macro_derive(NetSiteForceAndVirial)]
pub fn net_site_force_derive(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    net_site_force_and_virial::net_site_force_and_virial(input).into()
}

/// Automatically implement the `hoomd_interaction::NetSiteForceVirialAndTorque` trait.
///
/// The implemented `net_site_force_virial_and_torque` sums the result of `net_site_force_virial_and_torque`
/// over all fields.
///
/// Valid on:
/// * Structs with named fields.
/// * Tuple structs.
#[proc_macro_derive(NetSiteForceVirialAndTorque)]
pub fn net_site_force_and_torque_derive(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    net_site_force_virial_and_torque::net_site_force_virial_and_torque(input).into()
}

/// Automatically implement the `hoomd_microstate::property::NetTorque` trait.
///
/// The derived implementation returns a reference to the structure's `net_torque`
/// field.
///
/// Valid on structs with named fields.
#[proc_macro_derive(NetTorque)]
pub fn net_torque_derive(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    net_torque::net_torque(input)
}

/// Automatically implement the `hoomd_microstate::property::NetVirial` trait.
///
/// The derived implementation returns a reference to the structure's `net_virial`
/// field.
///
/// Valid on structs with named fields.
#[proc_macro_derive(NetVirial)]
pub fn net_virial_derive(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    net_virial::net_virial(input)
}

/// Automatically implement the `hoomd_microstate::property::Orientation` trait.
///
/// The derived implementation returns a reference to the structure's `orientation`
/// field.
///
/// Valid on structs with named fields.
#[proc_macro_derive(Orientation)]
pub fn orientation_derive(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    orientation::orientation(input)
}

/// Automatically implement the `hoomd_microstate::property::Position` trait.
///
/// The derived implementation returns a reference to the structure's `position`
/// field.
///
/// Valid on structs with named fields.
#[proc_macro_derive(Position)]
pub fn position_derive(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    position::position(input)
}

/// Automatically implement the `hoomd_microstate::property::RotationalDrag` trait.
///
/// The derived implementation returns a reference to the structure's `rotational_drag`
/// field.
///
/// Valid on structs with named fields.
#[proc_macro_derive(RotationalDrag)]
pub fn rotational_drag_derive(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    rotational_drag::rotational_drag(input)
}

/// Automatically implement the `hoomd_interaction::SitePairEnergy` trait.
///
/// The implemented `site_pair_energy` sums the result of `site_pair_energy`
/// over all fields. The implementation returns early when any one field returns
/// infinity. The implemented `site_pair_energy_initial` behaves similarly.
/// The derived `is_only_infinite_or_zero` returns true only when all fields
/// also return true for the same method.
///
/// Valid on:
/// * Structs with named fields.
/// * Tuple structs.
#[proc_macro_derive(SitePairEnergy)]
pub fn site_pair_energy_derive(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    site_pair_energy::site_pair_energy(input).into()
}

/// Automatically implement the `hoomd_interaction::TotalEnergy` trait.
///
/// The implemented `total_energy` sums the result of `total_energy`
/// over all fields. The implementation returns early when any one field returns
/// infinity.
///
/// Valid on:
/// * Structs with named fields.
/// * Tuple structs.
#[proc_macro_derive(TotalEnergy)]
pub fn total_energy_derive(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    total_energy::total_energy(input).into()
}

/// Automatically add all fields and implement all property traits for `hoomd_microstate::property::DynamicPoint`.
///
/// Arguments:
/// * `V`: The vector type to use for position, etc.  This type must implement
///   [`Default`], `hoomd_vector::Vector`, and `hoomd_vector::Outer`, and its
///   associated type through `Outer` must additionally implement [`Default`].
///   Example: `hoomd_vector::Cartesian::<2>`.
/// 
/// Valid on:
/// * Structs with named fields.
#[proc_macro_attribute]
pub fn derive_dynamic_point(input: TokenStream, annotated_item: TokenStream) -> TokenStream {
    // Parse the input to get the vector type. The type must implement `Outer`.
    let vector_type = parse_macro_input!(input as Type);
    
    // Parse the target struct, ensuring it has named fields
    let mut target_struct = parse_macro_input!(annotated_item as ItemStruct);
    let (impl_generics, ty_generics, where_clause) = target_struct.generics.split_for_impl();

    // Later on, an impl block for Transform<OrientedPoint> will require special
    // trait bounds
    let mut transform_generics = target_struct.generics.clone();
    transform_generics.params.push(parse_quote!(__R: Copy));
    let (transform_impl_generics, _, transform_where_clause) = transform_generics.split_for_impl();
    
    // Parse the original user fields
    let target_fields = match &mut target_struct.fields {
        Fields::Named(fields) => fields,
        Fields::Unnamed(_) | Fields::Unit => {
            return syn::Error::new_spanned(
                target_struct,
                "#[derive_dynamic_point(...)] only supports structs with named fields"
            )
            .to_compile_error()
            .into();
        }
    };
    
    // Add where-clause requirements
    let other_where: WhereClause = syn::parse_quote! {
        where
            #vector_type: Default + hoomd_vector::Outer + hoomd_vector::Vector,
            <#vector_type as hoomd_vector::Outer>::Tensor: Default
    };
    let final_where_clause = match where_clause {
        Some(original) => {
            let mut local_other_where = other_where.clone();
            local_other_where.predicates.extend(original.predicates.clone());
            local_other_where
        }
        None => other_where.clone()
    };
    let final_transform_where_clause = match transform_where_clause {
        Some(original) => {
            let mut local_other_where = other_where.clone();
            local_other_where.predicates.extend(original.predicates.clone());
            local_other_where
        }
        None => other_where
    };
    
    // Fields to add to the target struct
    let position: syn::Field = parse_quote! { pub position: #vector_type };
    let mass: syn::Field = parse_quote! { pub mass: f64 };
    let momentum: syn::Field = parse_quote! { pub momentum: #vector_type };
    let net_force: syn::Field = parse_quote! { pub net_force: #vector_type };
    let net_virial: syn::Field = parse_quote! { pub net_virial: <#vector_type as hoomd_vector::Outer>::Tensor };
    let drag: syn::Field = parse_quote! { pub drag: f64 };

    // Prepend the fields created above before the existing fields
    let old_fields = std::mem::take(&mut target_fields.named);

    let original_field_idents: Vec<_> = old_fields
        .iter()
        .map(|field| field.ident.as_ref().unwrap().clone())
        .collect();

    let original_field_types: Vec<_> = old_fields
        .iter()
        .map(|field| field.ty.clone())
        .collect();

    let original_default_values: Vec<syn::Expr> = original_field_types
        .iter()
        .map(|field_type| syn::parse_quote!( <#field_type as Default>::default() ))
        .collect();

    let mut new_fields = syn::punctuated::Punctuated::new();
    new_fields.push(position);
    new_fields.push(mass);
    new_fields.push(momentum);
    new_fields.push(net_force);
    new_fields.push(net_virial);
    new_fields.push(drag);

    for field in old_fields {
        new_fields.push(field);
    };

    target_fields.named = new_fields;

    // Add derive macro calls
    target_struct.attrs.push(syn::parse_quote! {
        #[derive(
            Clone, Copy, Debug, PartialEq,
            serde::Serialize,
            serde::Deserialize,
            hoomd_microstate::property::Position,
            hoomd_microstate::property::Mass,
            hoomd_microstate::property::Momentum,
            hoomd_microstate::property::NetForce,
            hoomd_microstate::property::NetVirial,
            hoomd_microstate::property::Drag,
        )]
    });

    // Get the name of the target struct
    let struct_name = &target_struct.ident;

    // Output the modified target struct
    TokenStream::from(quote! {
        #target_struct

        impl #impl_generics Default for #struct_name #ty_generics #final_where_clause {
            #[inline]
            fn default() -> Self {
                Self {
                    position: Default::default(),
                    mass: 1.0,
                    momentum: Default::default(),
                    net_force: Default::default(),
                    net_virial: <#vector_type as hoomd_vector::Outer>::Tensor::default(),
                    drag: 1.0,
                    #(#original_field_idents: #original_default_values),*
                }
            }
        }

        impl #impl_generics hoomd_microstate::Transform<hoomd_microstate::property::Point<#vector_type>> for #struct_name #ty_generics #final_where_clause {
            #[inline]
            fn transform(
                &self,
                site_properties: &hoomd_microstate::property::Point<#vector_type>
            ) -> hoomd_microstate::property::Point<#vector_type> {
                hoomd_microstate::property::Point {
                    position: self.position + site_properties.position,
                }
            }
        }

        impl #transform_impl_generics hoomd_microstate::Transform<hoomd_microstate::property::OrientedPoint<#vector_type, __R>> for #struct_name #ty_generics #final_transform_where_clause {
            #[inline]
            fn transform(
                &self,
                site_properties: &hoomd_microstate::property::OrientedPoint<#vector_type, __R>
            ) -> hoomd_microstate::property::OrientedPoint<#vector_type, __R> {
                hoomd_microstate::property::OrientedPoint {
                    position: self.position + site_properties.position,
                    ..*site_properties
                }
            }
        }
    })
}

/// Automatically add all fields and implement all property traits for `hoomd_microstate::property::DynamicOrientedPoint`.
///
/// Arguments:
/// * `V`: The vector type to use for position, etc. This type must implement
///   [`Default`], `hoomd_vector::Vector`, `hoomd_vector::Outer`, and
///   `hoomd_vector::Wedge`, and its associated types through `Outer` and
///   `Wedge` must additionally implement [`Default`]. Example:
///   `hoomd_vector::Cartesian::<2>`.
/// * `R`: The rotation type to use for orientation. This type must implement
///   `hoomd_microstate::property::RotationalMotionTypes`,
///   `hoomd_vector::Rotate<V>` and `hoomd_vector::Rotation`. Example:
///   `hoomd_vector::Angle`.
/// 
/// Valid on:
/// * Structs with named fields.
#[proc_macro_attribute]
pub fn derive_dynamic_oriented_point(input: TokenStream, annotated_item: TokenStream) -> TokenStream {
    // Parse the input to get the vector and rotation types.
    struct TypePair {
        first: Type,
        _comma: Token![,],
        second: Type,
    }

    impl Parse for TypePair {
        fn parse(input: ParseStream) -> Result<Self, syn::Error> {
            Ok(TypePair {
                first: input.parse()?,
                _comma: input.parse()?,
                second: input.parse()?,
            })
        }
    }
    let input_types = parse_macro_input!(input as TypePair);
    let vector_type = input_types.first;
    let rotation_type = input_types.second;
    
    // Parse the target struct, ensuring it has named fields
    let mut target_struct = parse_macro_input!(annotated_item as ItemStruct);
    let (impl_generics, ty_generics, where_clause) = target_struct.generics.split_for_impl();

    // Parse the original user fields
    let target_fields = match &mut target_struct.fields {
        Fields::Named(fields) => fields,
        Fields::Unnamed(_) | Fields::Unit => {
            return syn::Error::new_spanned(
                target_struct,
                "#[derive_dynamic_oriented_point(...)] only supports structs with named fields"
            )
            .to_compile_error()
            .into();
        }
    };
    
    // Add where-clause requirements
    let mut other_where: WhereClause = syn::parse_quote! {
        where
            #vector_type: Default
                + hoomd_vector::Outer
                + hoomd_vector::Wedge
                + hoomd_vector::Vector,
            <#vector_type as hoomd_vector::Outer>::Tensor: Default,
            <#vector_type as hoomd_vector::Wedge>::Bivector: Default,
            #rotation_type: hoomd_microstate::property::RotationalMotionTypes
                + hoomd_vector::Rotate<#vector_type>
                + hoomd_vector::Rotation,
    };
    let final_where_clause = match where_clause {
        Some(original) => {
            other_where.predicates.extend(original.predicates.clone());
            other_where
        }
        None => other_where
    };
    
    // Fields to add to the target struct
    let position: syn::Field = parse_quote! { pub position: #vector_type };
    let orientation: syn::Field = parse_quote! { pub orientation: #rotation_type };
    let mass: syn::Field = parse_quote! { pub mass: f64 };
    let momentum: syn::Field = parse_quote! { pub momentum: #vector_type };
    let net_force: syn::Field = parse_quote! { pub net_force: #vector_type };
    let net_virial: syn::Field = parse_quote! { pub net_virial: <#vector_type as hoomd_vector::Outer>::Tensor };
    let moment_of_inertia: syn::Field = parse_quote! { pub moment_of_inertia: <#rotation_type as hoomd_microstate::property::RotationalMotionTypes>::MomentOfInertia };
    let angular_momentum: syn::Field = parse_quote! { pub angular_momentum: <#rotation_type as hoomd_microstate::property::RotationalMotionTypes>::AngularMomentum };
    let net_torque: syn::Field = parse_quote! { pub net_torque: <#vector_type as hoomd_vector::Wedge>::Bivector };
    let drag: syn::Field = parse_quote! { pub drag: f64 };
    let rotational_drag: syn::Field = parse_quote! { pub rotational_drag: <#rotation_type as hoomd_microstate::property::RotationalMotionTypes>::RotationalDrag };

    // Prepend the fields created above before the existing fields
    let old_fields = std::mem::take(&mut target_fields.named);

    let original_field_idents: Vec<_> = old_fields
        .iter()
        .map(|field| field.ident.as_ref().unwrap().clone())
        .collect();

    let original_field_types: Vec<_> = old_fields
        .iter()
        .map(|field| field.ty.clone())
        .collect();

    let original_default_values: Vec<syn::Expr> = original_field_types
        .iter()
        .map(|field_type| syn::parse_quote!( <#field_type as Default>::default() ))
        .collect();

    let mut new_fields = syn::punctuated::Punctuated::new();
    new_fields.push(position);
    new_fields.push(orientation);
    new_fields.push(mass);
    new_fields.push(momentum);
    new_fields.push(net_force);
    new_fields.push(net_virial);
    new_fields.push(moment_of_inertia);
    new_fields.push(angular_momentum);
    new_fields.push(net_torque);
    new_fields.push(drag);
    new_fields.push(rotational_drag);

    for field in old_fields {
        new_fields.push(field);
    };

    target_fields.named = new_fields;

    // Add derive macro calls
    target_struct.attrs.push(syn::parse_quote! {
        #[derive(
            Clone, Copy, Debug, PartialEq,
            serde::Serialize,
            serde::Deserialize,
            hoomd_microstate::property::Position,
            hoomd_microstate::property::Orientation,
            hoomd_microstate::property::Mass,
            hoomd_microstate::property::Momentum,
            hoomd_microstate::property::NetForce,
            hoomd_microstate::property::NetVirial,
            hoomd_microstate::property::MomentOfInertia,
            hoomd_microstate::property::AngularMomentum,
            hoomd_microstate::property::NetTorque,
            hoomd_microstate::property::Drag,
            hoomd_microstate::property::RotationalDrag,
        )]
    });

    // Get the name of the target struct
    let struct_name = &target_struct.ident;

    // Output the modified target struct
    TokenStream::from(quote! {
        #target_struct

        impl #impl_generics Default for #struct_name #ty_generics #final_where_clause {
            #[inline]
            fn default() -> Self {
                Self {
                    position: Default::default(),
                    orientation: Default::default(),
                    mass: 1.0,
                    momentum: Default::default(),
                    net_force: Default::default(),
                    net_virial: <#vector_type as hoomd_vector::Outer>::Tensor::default(),
                    moment_of_inertia: <#rotation_type as hoomd_microstate::property::RotationalMotionTypes>::default_moment_of_inertia(),
                    angular_momentum: Default::default(),
                    net_torque: Default::default(),
                    drag: 1.0,
                    rotational_drag: <#rotation_type as hoomd_microstate::property::RotationalMotionTypes>::default_rotational_drag(),
                    #(#original_field_idents: #original_default_values),*
                }
            }
        }

        impl #impl_generics hoomd_microstate::Transform<hoomd_microstate::property::Point<#vector_type>> for #struct_name #ty_generics #final_where_clause {
            #[inline]
            fn transform(
                &self,
                site_properties: &hoomd_microstate::property::Point<#vector_type>
            ) -> hoomd_microstate::property::Point<#vector_type> {
                hoomd_microstate::property::Point {
                    position:
                        self.position
                        + <#rotation_type as hoomd_vector::Rotate<#vector_type>>::rotate(
                            <Self as hoomd_microstate::property::Orientation>::orientation(&self),
                            &site_properties.position
                        ),
                }
            }
        }

        impl #impl_generics hoomd_microstate::Transform<hoomd_microstate::property::OrientedPoint<#vector_type, #rotation_type>> for #struct_name #ty_generics #final_where_clause {
            #[inline]
            fn transform(
                &self,
                site_properties: &hoomd_microstate::property::OrientedPoint<#vector_type, #rotation_type>
            ) -> hoomd_microstate::property::OrientedPoint<#vector_type, #rotation_type> {
                hoomd_microstate::property::OrientedPoint {
                    position:
                        self.position
                        + <#rotation_type as hoomd_vector::Rotate<#vector_type>>::rotate(
                            <Self as hoomd_microstate::property::Orientation>::orientation(&self),
                            &site_properties.position
                        ),
                    orientation:
                        <#rotation_type as hoomd_vector::Rotation>::combine(
                            <Self as hoomd_microstate::property::Orientation>::orientation(&self),
                            &site_properties.orientation
                        ),
                }
            }
        }
    })
}