# hoomd_utility

## Value validation

The `hoomd_utility::valid` module implements a number of newtypes that ensure values are
given in proper ranges. These implement `try_from` to allow simple conversion from
native types. All documentation examples should encourage `try_into` syntax:
```
let v = 1.0;
let validated = Valid(v.try_into()?);
```
When using try_into, there is no need for the user to `use` the validator type.

All validator newtypes should have private members and provide read-only get methods.
