"""Knowledge-graph schema for the catalyst-kg-agent.

Design notes
------------
- pydantic v2 BaseModel for every node and edge -- single source of truth for
  validation, JSON-serializability, and shared vocabulary with `agent/schemas.py`.
- Stable, namespaced string IDs (`material:mp-126`, `element:Ni`,
  `chemsys:Ni-P`, `structure:mp-126`, `property:mp-126:band_gap`). Keeps the
  graph human-debuggable in NetworkX / gephi and round-trip-safe.
- Node types kept narrow on purpose: Element is a real node (not just an
  attribute on Material) so chemsys-level queries and provenance
  ("which materials contain Ni?") are first-class traversals, not
  string filtering.
- Property is a node, not an attribute, so that predicted vs measured
  properties can coexist on the same Material node and provenance
  (which model / which source) is preserved on the edge.
- Units are explicit strings, not magic floats. eV vs eV/atom is a
  historically expensive bug to debug downstream.
- The schema module owns no I/O. `kg/build_graph.py` is the only place
  that reads CIFs / metadata.json; this module only defines shapes.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Canonical vocab (enums)
# ---------------------------------------------------------------------------

class NodeType(str, Enum):
    """Canonical node types. Used as both NetworkX node attribute and as
    the discriminator key when (de)serializing typed sub-models."""
    MATERIAL = "Material"
    ELEMENT = "Element"
    CHEMSYS = "Chemsys"
    STRUCTURE = "Structure"
    PROPERTY = "Property"


class EdgeType(str, Enum):
    """Canonical edge (relation) types. Predicates are short CamelCase verbs,
    matching typical RDF/OWL convention."""
    HAS_ELEMENT = "HAS_ELEMENT"      # Material -> Element
    IN_CHEMSYS = "IN_CHEMSYS"        # Material -> Chemsys
    HAS_STRUCTURE = "HAS_STRUCTURE"  # Material -> Structure
    HAS_PROPERTY = "HAS_PROPERTY"    # Material -> Property


class PropertyName(str, Enum):
    """Property name vocabulary. The enum is the *canonical* set; new names
    can be added here explicitly. The `PropertyName.is_canonical(name)`
    helper and `PropertyName.coerce(name)` constructor let `build_graph.py`
    accept either an enum member or a raw string from a future MP endpoint,
    so adding a new property to the schema is a one-line change here and
    does not require touching the property-bearing graph code."""
    ENERGY_ABOVE_HULL = "energy_above_hull"
    BAND_GAP = "band_gap"

    @classmethod
    def is_canonical(cls, name: object) -> bool:
        return isinstance(name, cls) or (
            isinstance(name, str) and name in cls._value2member_map_
        )

    @classmethod
    def coerce(cls, name: object) -> "PropertyName":
        """Return the enum member for `name` if it matches a canonical value;
        otherwise return a synthetic member whose value is the raw string.

        Synthetic members are *not* registered in `_value2member_map_`, so
        `is_canonical` will still return False for them. This keeps the
        canonical surface auditable while letting the graph ingest
        arbitrary MP property fields without a schema edit per property.
        """
        if isinstance(name, cls):
            return name
        if isinstance(name, str) and name in cls._value2member_map_:
            return cls._value2member_map_[name]
        # Synthetic: bypass Enum.__init__ to avoid mutating the canonical set.
        synthetic = str.__new__(cls, name)
        synthetic._name_ = name  # type: ignore[attr-defined]
        synthetic._value_ = name  # type: ignore[attr-defined]
        return synthetic


class PropertyUnit(str, Enum):
    ENERGY_ABOVE_HULL = "eV/atom"
    BAND_GAP = "eV"

    @classmethod
    def coerce(cls, unit: object) -> "PropertyUnit":
        if isinstance(unit, cls):
            return unit
        if isinstance(unit, str) and unit in cls._value2member_map_:
            return cls._value2member_map_[unit]
        synthetic = str.__new__(cls, unit)
        synthetic._name_ = unit  # type: ignore[attr-defined]
        synthetic._value_ = unit  # type: ignore[attr-defined]
        return synthetic


class PropertySource(str, Enum):
    """Where the property value came from. Critical for FAIR provenance
    and for keeping UMA/OMat24-derived values in a separate tier
    (per PROJECT_STATE: 'never numerically mixed with MP-derived stats')."""
    MATERIALS_PROJECT = "MaterialsProject"
    MACE_FINETUNED = "MACE-finetuned"
    CGCNN_BASELINE = "CGCNN-baseline"
    UMA_RELAX = "UMA-relax"
    UNKNOWN = "Unknown"

    @classmethod
    def coerce(cls, source: object) -> "PropertySource":
        if isinstance(source, cls):
            return source
        if isinstance(source, str) and source in cls._value2member_map_:
            return cls._value2member_map_[source]
        # Unknown new source: keep the raw label, do not pollute the canonical set.
        synthetic = str.__new__(cls, source)
        synthetic._name_ = source  # type: ignore[attr-defined]
        synthetic._value_ = source  # type: ignore[attr-defined]
        return synthetic


class CrystalSystem(str, Enum):
    """7 crystal systems per ITCA. pymatgen's Structure.get_crystal_system()
    returns these as strings; we coerce to the enum for schema safety."""
    TRICLINIC = "triclinic"
    MONOCLINIC = "monoclinic"
    ORTHORHOMBIC = "orthorhombic"
    TETRAGONAL = "tetragonal"
    TRIGONAL = "trigonal"
    HEXAGONAL = "hexagonal"
    CUBIC = "cubic"


# ---------------------------------------------------------------------------
# ID helpers
# ---------------------------------------------------------------------------

def material_id(mpid: str) -> str:
    return f"material:{mpid}"


def element_id(symbol: str) -> str:
    return f"element:{symbol}"


def chemsys_id(symbols: list[str]) -> str:
    """Chemsys IDs are sorted to make 'Ni-P' and 'P-Ni' collapse to one node."""
    return "chemsys:" + "-".join(sorted(symbols))


def structure_id(mpid: str) -> str:
    return f"structure:{mpid}"


def property_id(mpid: str, name: PropertyName | str) -> str:
    """Build a PropertyNode ID. Accepts either an enum member or a raw
    string (the latter is the open-vocabulary case for future MP fields
    not yet in the canonical PropertyName enum)."""
    raw = name.value if isinstance(name, PropertyName) else str(name)
    return f"property:{mpid}:{raw}"


# ---------------------------------------------------------------------------
# Node models
# ---------------------------------------------------------------------------

class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=False, str_strip_whitespace=True)


class ElementNode(_Base):
    """One per chemical element symbol. Singleton within a built graph."""
    id: str = Field(..., description="element:<Symbol>, e.g. 'element:Ni'")
    type: NodeType = NodeType.ELEMENT
    symbol: str = Field(..., min_length=1, max_length=2)
    # Name left out -- pymatgen/mendeleev lookup belongs in build_graph, not schema.

    @field_validator("symbol")
    @classmethod
    def _cap(cls, v: str) -> str:
        # Element symbols are case-sensitive (Co vs CO); canonicalize first letter only.
        if not v:
            raise ValueError("empty element symbol")
        return v[0].upper() + v[1:].lower() if len(v) > 1 else v.upper()


class ChemsysNode(_Base):
    """A unique set of elements defining a chemical system. Sort order is
    canonicalized so {Ni,P} and {P,Ni} map to the same node."""
    id: str = Field(..., description="chemsys:<sorted-dash-joined symbols>")
    type: NodeType = NodeType.CHEMSYS
    symbols: list[str] = Field(..., min_length=1)

    @field_validator("symbols")
    @classmethod
    def _sorted_unique(cls, v: list[str]) -> list[str]:
        # Sort by symbol so two {Ni,P} inputs collapse to one node.
        return sorted(set(v))


class StructureNode(_Base):
    """Crystal structure (CIF-derived). 1:1 with a material_id in the current
    data flow, but kept separate so future sources (slabs, adsorbates,
    OQMD) can attach alternative structures to the same Material."""
    id: str = Field(..., description="structure:<mpid>")
    type: NodeType = NodeType.STRUCTURE
    mpid: str = Field(..., description="MP material_id this structure belongs to")
    cif_path: str = Field(..., description="Path relative to repo root, e.g. 'data/raw/structures/mp-126.cif'")
    space_group_symbol: Optional[str] = None
    space_group_number: Optional[int] = Field(default=None, ge=1, le=230)
    crystal_system: Optional[CrystalSystem] = None
    num_sites: Optional[int] = Field(default=None, ge=1)


class PropertyNode(_Base):
    """A measured or predicted scalar property attached to one Material.
    Source matters: see PropertySource for the tier-separation rule.

    `name` is *open*: canonical PropertyName enum members are accepted
    directly, and any new string from a future MP endpoint is coerced
    via `PropertyName.coerce` so the graph does not require a schema
    edit per new propesource", mode="before")
    @classmethod
    def _accept_open_source(cls, v: object) -> PropertySource:
        if isinstance(v, PropertySource):
            return v
        if isinstance(v, str):
            return PropertySource.coerce(v)
        raise ValueError(f"PropertyNode.source must be str or PropertySource, got {type(v).__name__}")

    @field_validator("rty. Use `PropertyName.is_canonical(name)` to
    audit whether a given property is on the canonical list.
    """
    id: str = Field(..., description="property:<mpid>:<name>")
    type: NodeType = NodeType.PROPERTY
    mpid: str
    name: PropertyName
    value: float
    unit: PropertyUnit
    source: PropertySource = PropertySource.UNKNOWN

    @field_validator("name", mode="before")
    @classmethod
    def _accept_open(cls, v: object) -> PropertyName:
        if isinstance(v, PropertyName):
            return v
        if isinstance(v, str):
            return PropertyName.coerce(v)
        raise ValueError(f"PropertyNode.name must be str or PropertyName, got {type(v).__name__}")

    @field_validator("unit", mode="before")
    @classmethod
    def _accept_open_unit(cls, v: object) -> PropertyUnit:
        # Mirrors the `name` open-vocabulary policy. New MP properties
        # arrive with units not yet in the canonical enum; coerce so the
        # graph stays ingestible without a per-unit schema edit.
        if isinstance(v, PropertyUnit):
            return v
        if isinstance(v, str):
            return PropertyUnit.coerce(v)
        raise ValueError(f"PropertyNode.unit must be str or PropertyUnit, got {type(v).__name__}")


class MaterialNode(_Base):
    """Top-level discovery target. One per material_id.

    Contains an optional reference to its StructureNode via ID (not the full
    Structure object). This keeps the schema simple and allows multiple
    structures (slabs, adsorbates) to be linked to one Material."""
    id: str = Field(..., description="material:<mpid>")
    type: NodeType = NodeType.MATERIAL
    mpid: str
    formula_pretty: str
    elements: list[str] = Field(..., min_length=1)
    structure_id: Optional[str] = Field(
        default=None,
        description="Optional reference to StructureNode ID (e.g. 'structure:mp-126')"
    )

    @field_validator("elements")
    @classmethod
    def _clean(cls, v: list[str]) -> list[str]:
        # pymatgen sometimes returns element objects serialized as 'Element'
        # repr; the download script already casts to str, but defend anyway.
        return [str(e) for e in v]


# ---------------------------------------------------------------------------
# Edge models
# ---------------------------------------------------------------------------

class _EdgeBase(_Base):
    """Edges carry provenance + optional numeric attributes. Multi-edges
    (Material --HAS_ELEMENT--> Element, with a count) require
    `key=` on add_edge; we store the count on the edge attribute dict
    keyed by this edge's id."""
    id: str
    type: EdgeType
    source_id: str
    target_id: str


class HasElementEdge(_EdgeBase):
    type: EdgeType = EdgeType.HAS_ELEMENT
    count: int = Field(..., ge=1, description="Number of atoms of this element in the formula unit")


class InChemsysEdge(_EdgeBase):
    type: EdgeType = EdgeType.IN_CHEMSYS


class HasStructureEdge(_EdgeBase):
    type: EdgeType = EdgeType.HAS_STRUCTURE


class HasPropertyEdge(_EdgeBase):
    type: EdgeType = EdgeType.HAS_PROPERTY
    # The Property node already carries value/unit/source; the edge stays
    # minimal so we don't double-record. Confidence / uncertainty can be
    # added here later without touching the Property node shape.


# ---------------------------------------------------------------------------
# Type aliases for code that wants the union without enumerating
# ---------------------------------------------------------------------------

KGNode = ElementNode | ChemsysNode | StructureNode | PropertyNode | MaterialNode
KGEdge = HasElementEdge | InChemsysEdge | HasStructureEdge | HasPropertyEdge


__all__ = [
    # enums
    "NodeType", "EdgeType", "PropertyName", "PropertyUnit", "PropertySource",
    "CrystalSystem",
    # node models
    "ElementNode", "ChemsysNode", "StructureNode", "PropertyNode", "MaterialNode",
    # edge models
    "HasElementEdge", "InChemsysEdge", "HasStructureEdge", "HasPropertyEdge",
    # unions
    "KGNode", "KGEdge",
    # id helpers
    "material_id", "element_id", "chemsys_id", "structure_id", "property_id",
]
