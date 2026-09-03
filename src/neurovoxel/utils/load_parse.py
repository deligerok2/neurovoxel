"""Data loader and config validation for NeuroVoxel app."""

# pyright: reportMissingTypeStubs=false

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import bids2table as b2t2
import bidsschematools as bst
import jsonschema
import pandas as pd
from formulaic import (
    model_matrix,  # pyright: ignore[reportUnknownVariableType]
)

from neurovoxel.utils import SCHEMA

if TYPE_CHECKING:
    from bidsschematools.types import Namespace  # type: ignore  # noqa: PGH003


def load_config(config_file: Path) -> dict[str, Any]:
    """Load and validate a NeuroVoxel configuration file."""
    with config_file.open("r") as f:
        config = json.load(f)

    jsonschema.validate(config, SCHEMA)

    return config


def config_to_schema(config: Path) -> Namespace:
    """Convert a NeuroVoxel config file to a Namespace object."""
    schema = bst.schema.load_schema()  # type: ignore  # noqa: PGH003
    with config.open("r") as f:
        config_dict = json.load(f)["entities"]
        config_dict = [  # this removes the original
            item for item in config_dict if item.get("name") != "trc"
        ]
        for i in config_dict:
            if i["name"] not in schema.objects.entities:  # type: ignore  # noqa: PGH003
                schema.objects.entities[i["name"]] = {  # type: ignore  # noqa: PGH003
                    "display_name": i["name"],
                    "description": i["name"],
                    "name": i["name"],
                    "type": "string",
                    "format": "label",
                }
            if i["name"] not in schema.rules.entities:  # pyright: ignore[reportUnknownMemberType]
                schema.rules.entities.append(i["name"])  # pyright: ignore[reportUnknownMemberType]

    return schema  # type: ignore[return-value]


def load_bids(
    bids_root: Path,
    schema: Namespace | None = None,
) -> pd.DataFrame:
    """Load BIDS dataset."""
    layout = b2t2.index_dataset(  # pyright: ignore[reportUnknownVariableType] # pyright: ignore[reportUnknownMemberType] # type: ignore  # noqa: PGH003
        bids_root,
        schema=schema,
    ).to_pandas()
    layout["nifti_files"] = layout.apply(  # pyright: ignore[reportUnknownMemberType]
        lambda row: str(Path(row["root"]) / row["path"]),  # type: ignore  # noqa: PGH003
        axis=1,
    )
    return layout  # pyright: ignore[reportUnknownVariableType]


def parse_layout(table: pd.DataFrame) -> pd.DataFrame:
    """Create a table of unique imaging types from a bids2table DataFrame."""
    # Keep only NIfTI files
    entity_df = table[
        table["ext"].isin([".nii.gz", ".nii"])  # pyright: ignore[reportUnknownMemberType]
    ].copy()
    # Columns that define a unique imaging type
    entity_cols = [
        "datatype",
        "desc",
        "space",
        "suffix",
        "param",
        "meas",
        "trc",
    ]

    # Keep only columns that actually exist
    entity_cols = [col for col in entity_cols if col in entity_df.columns]

    # Keep only those columns
    entity_df = entity_df[entity_cols]

    # Remove duplicate ROWS
    entity_df = entity_df.drop_duplicates().reset_index(drop=True)

    # Sort
    sort_cols = [
        col
        for col in ["datatype", "suffix", "desc", "param"]
        if col in entity_df.columns
    ]

    entity_df = entity_df.sort_values(by=sort_cols).reset_index(drop=True)

    # Create user-facing name
    def concat_name(row: pd.Series) -> str:
        parts = [
            str(row[col])
            for col in ["desc", "param", "trc", "meas", "suffix"]
            if col in row and pd.notna(row[col])
        ]
        return "_".join(parts) if parts else "Enter name here"

    entity_df["name"] = entity_df.apply(concat_name, axis=1)

    # Put name first
    entity_df = entity_df[["name", *entity_cols]]

    return entity_df


def parse_query(
    query: str,
    allowed_lhs_values: list[str],
    rhs_df: pd.DataFrame,
) -> tuple[str, pd.Index]:
    """Parse query to extract the left and right hand side."""
    if "~" in query:
        lhs, rhs = query.split("~")
        lhs = lhs.strip()
        rhs = rhs.strip()
        if lhs not in allowed_lhs_values:
            msg = "Imaging outcome in query is invalid"
            raise ValueError(msg)
        x_mat = model_matrix(rhs, rhs_df)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]
        return lhs, x_mat.columns  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]

    msg = "Invalid formula syntax: formula should contain '~'"
    raise ValueError(msg)
