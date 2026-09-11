"""Display data table."""

import pandas as pd
import streamlit as st
from pandas import DataFrame


def render_entity_table(entity_df: DataFrame) -> DataFrame:
    """Editable table: only 'name' is editable; duplicate names are disambiguated."""
    st.write("Types of images in dataset:")

    # Make only the 'name' column editable
    disabled_cols = [c for c in entity_df.columns if c != "name"]

    cols = list(entity_df.columns)
    if "name" in cols:
        cols.remove("name")
        cols = ["name", *cols]

    # Find duplicated names
    duplicated_names = entity_df.loc[
        entity_df["name"].duplicated(keep=False), "name"
    ].unique()

    if len(duplicated_names) > 0:
        for name in duplicated_names:
            mask = entity_df["name"] == name
            duplicate_rows = entity_df.loc[mask] # type: ignore  # noqa: PGH003

            for idx, row in duplicate_rows.iterrows(): # type: ignore  # noqa: PGH003
                # Use every column except name to disambiguate
                suffix_values = [
                    str(row[col]) # type: ignore  # noqa: PGH003
                    for col in entity_df.columns
                     if col != "name" and pd.notna(row[col])  # type: ignore  # noqa: PGH003
                     and str(row[col]) != str(name) # type: ignore  # noqa: PGH003
                ]

                entity_df.loc[idx, "name"] = f"{name}_{'_'.join(suffix_values)}"

    edited_df = st.data_editor(
            entity_df,
            disabled=disabled_cols,
            key="entity_table_editor",
            width="content",
            column_order=cols,
            hide_index=True,
        )

    # Check whether the resulting names are unique
    if edited_df["name"].duplicated().any():
        st.error(
            "Entries in the 'name' column must be unique. "
            "Please fix duplicates before continuing."
        )

    return edited_df
