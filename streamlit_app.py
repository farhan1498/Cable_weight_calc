"""
Streamlit Cable Weight Calculator - Interactive Web Application
Comprehensive tool for calculating submarine cable weights in air and seawater
"""
import streamlit as st
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import pandas as pd


@dataclass
class CableSpecs:
    """Cable specification parameters"""
    # Conductor specifications
    conductor_size_mm2: float = 1200
    conductor_material: str = "Al"  # "Al" or "Cu"

    # Layer specifications
    inner_semicon_od_mm: float = 47
    inner_semicon_thickness_mm: float = 2
    xlpe_od_mm: float = 98
    outer_semicon_od_mm: float = 101
    water_block_tape_thickness_mm: float = 1.2
    water_block_tape_od_mm: float = 103
    copper_screen_size_mm2: float = 0
    lead_sheath_thickness_mm: float = 2.3
    lead_sheath_od_mm: float = 108
    pe_sheath_thickness_mm: float = 3
    od_power_core_mm: float = 114

    # Cable assembly
    lay_length_mm: float = 4000

    # FOC specifications
    od_foc_mm: float = 12
    weight_single_foc_kg_per_m: float = 0.25
    num_focs: int = 2

    # Assembly and filler
    assembled_od_mm: float = 246
    filler_coverage_percent: float = 0.7
    filler_material: str = "Filler_PVC"

    # Bedding and armor
    bedding_od_mm: float = 251
    bedding_thickness_mm: float = 2.2
    num_steel_wires: int = 91
    num_pe_wires: int = 0
    wire_diameter_mm: float = 8
    armor_lay_length_mm: float = 4100
    steel_type: str = "GS"  # "GS" or "SS"

    # Outer protection
    outer_od_mm: float = 267
    outer_thickness_mm: float = 4.4
    design_type: str = "ROVED"  # "ROVED" or "SHEATHED"


class CableWeightCalculator:
    """Comprehensive cable weight calculator for submarine power cables"""

    def __init__(self, specs: Optional[CableSpecs] = None, show_breakdown: bool = True):
        self.specs = specs or CableSpecs()
        self.show_breakdown = show_breakdown
        self._material_data = self._init_materials()
        self._results = {}

    def _init_materials(self) -> Dict:
        """Initialize material density data"""
        raw_data = {
            "Seawater": (1025, 0),
            "Aluminium": (2710, 1685),
            "Copper": (8940, 7915),
            "Lead": (11370, 10345),
            "Galvanised Steel Wires (GS)": (7890, 6865),
            "Stainless Steel Wires (SS)": (7850, 6825),
            "Conductor Strandblock Water Blocking Material": (1170, 145),
            "XLPE": (920, -105),
            "Semi Con for XLPE": (1120, 95),
            "Semi Con Water Blocking Tape": (500, -525),
            "MDPE": (930, -95),
            "Semi Con PE": (1055, 30),
            "HDPE": (960, -65),
            "Polypropylene": (905, -120),
            "Bitumen": (1280, 255),
            "Filler_MDPE": (930, -95),
            "Filler_SC PE": (1055, 30),
            "Filler_PVC": (1380, 355),
        }

        materials = {}
        for material, (air, seawater) in raw_data.items():
            sg = air / (air - seawater) if air != seawater else None
            materials[material] = {
                "density_air": air,
                "density_seawater": seawater,
                "specific_gravity": round(sg, 3) if sg else None
            }
        return materials

    def calculate_layup_geometry(self) -> Dict:
        """Calculate 3-core layup geometry and cabling factors"""
        lay_len = self.specs.lay_length_mm
        od_core = self.specs.od_power_core_mm

        # 3-core layup radius (equilateral triangle)
        r_3core = (od_core / 2) / math.sin(math.radians(60))
        od_calc = 2 * r_3core + od_core
        circumference = 2 * math.pi * r_3core
        helix_len = math.sqrt(lay_len ** 2 + circumference ** 2)
        cabling_factor = helix_len / lay_len
        angle_deg = math.degrees(math.acos(lay_len / helix_len))

        return {
            "radius_3core_layup": round(r_3core, 4),
            "od_calculated": round(od_calc, 4),
            "circumference": round(circumference, 4),
            "helix_length": round(helix_len, 4),
            "cabling_factor": round(cabling_factor, 5),
            "layup_angle_deg": round(angle_deg, 2)
        }

    def calculate_conductor_weight(self, cabling_factor: float) -> float:
        """Calculate conductor weight in air"""
        material_key = "Aluminium" if self.specs.conductor_material == "Al" else "Copper"
        density = self._material_data[material_key]["density_air"]

        weight = (self.specs.conductor_size_mm2 * density * 1e-6 *
                  cabling_factor * 1.01)  # 1.01 = CR pass tolerance
        return round(weight, 3)

    def calculate_layer_weight(self, inner_d: float, outer_d: float,
                               material: str, fill_factor: float = 1.0) -> float:
        """Generic layer weight calculation"""
        density = self._material_data[material]["density_air"]
        area = math.pi * ((outer_d / 2) ** 2 - (inner_d / 2) ** 2)
        return round(area * density * 1e-6 * fill_factor, 3)

    def calculate_conductor_waterblock(self) -> Dict:
        """Calculate conductor waterblocking material weight"""
        density = self._material_data["Conductor Strandblock Water Blocking Material"]["density_air"]
        radius = (self.specs.inner_semicon_od_mm - 2 * self.specs.inner_semicon_thickness_mm) / 2
        vol_est = (math.pi * radius ** 2) - self.specs.conductor_size_mm2
        weight = vol_est * density * 1e-6

        return {
            "estimated_volume_mm2": round(vol_est, 3),
            "weight_in_air_kg_per_m": round(weight, 3)
        }

    def calculate_copper_screen(self) -> Dict:
        """Calculate copper screen wire weight with lay correction"""
        if self.specs.copper_screen_size_mm2 == 0:
            return {"adjusted_area_mm2": 0, "weight_in_air_kg_per_m": 0}

        density = self._material_data["Copper"]["density_air"]
        lay_angle = 21  # Standard lay angle
        adj_area = self.specs.copper_screen_size_mm2 / math.cos(math.radians(lay_angle))
        weight = adj_area * density * 1e-6

        return {
            "adjusted_area_mm2": round(adj_area, 3),
            "weight_in_air_kg_per_m": round(weight, 3)
        }

    def calculate_power_core(self) -> Dict:
        """Calculate complete power core weight and properties"""
        layup = self.calculate_layup_geometry()
        cabling_factor = layup["cabling_factor"]

        # Individual component weights
        conductor_wt = self.calculate_conductor_weight(cabling_factor)
        waterblock = self.calculate_conductor_waterblock()

        # Layer weights
        inner_semicon_wt = self.calculate_layer_weight(
            self.specs.inner_semicon_od_mm - 2 * self.specs.inner_semicon_thickness_mm,
            self.specs.inner_semicon_od_mm, "Semi Con for XLPE"
        )

        xlpe_wt = self.calculate_layer_weight(
            self.specs.inner_semicon_od_mm, self.specs.xlpe_od_mm, "XLPE"
        )

        outer_semicon_wt = self.calculate_layer_weight(
            self.specs.xlpe_od_mm, self.specs.outer_semicon_od_mm, "Semi Con for XLPE"
        )

        water_tape_wt = self.calculate_layer_weight(
            self.specs.water_block_tape_od_mm - 2 * self.specs.water_block_tape_thickness_mm,
            self.specs.water_block_tape_od_mm, "Semi Con Water Blocking Tape"
        )

        copper_screen = self.calculate_copper_screen()

        lead_wt = self.calculate_layer_weight(
            self.specs.lead_sheath_od_mm - 2 * self.specs.lead_sheath_thickness_mm,
            self.specs.lead_sheath_od_mm, "Lead"
        )

        pe_sheath_wt = self.calculate_layer_weight(
            self.specs.od_power_core_mm - 2 * self.specs.pe_sheath_thickness_mm,
            self.specs.od_power_core_mm, "Semi Con PE"
        )

        # Total weight calculation
        components = [
            conductor_wt, waterblock["weight_in_air_kg_per_m"],
            inner_semicon_wt, xlpe_wt, outer_semicon_wt,
            water_tape_wt, copper_screen["weight_in_air_kg_per_m"],
            lead_wt, pe_sheath_wt
        ]

        total_wt = sum(components)
        radius_m = (self.specs.od_power_core_mm / 2) * 1e-3
        area_m2 = math.pi * radius_m ** 2

        seawater_density = self._material_data["Seawater"]["density_air"]
        displaced_wt = seawater_density * area_m2
        seawater_wt = total_wt - displaced_wt
        overall_density = total_wt / area_m2

        return {
            "layup_geometry": layup,
            "component_weights": {
                "conductor": conductor_wt,
                "waterblock": waterblock["weight_in_air_kg_per_m"],
                "inner_semicon": inner_semicon_wt,
                "xlpe": xlpe_wt,
                "outer_semicon": outer_semicon_wt,
                "water_tape": water_tape_wt,
                "copper_screen": copper_screen["weight_in_air_kg_per_m"],
                "lead_sheath": lead_wt,
                "pe_sheath": pe_sheath_wt
            },
            "total_weight_kg_per_m": round(total_wt, 3),
            "overall_density_kg_per_m3": round(overall_density, 3),
            "weight_in_seawater_kg_per_m": round(seawater_wt, 3)
        }

    def calculate_foc_weight(self) -> Dict:
        """Calculate fiber optic cable weights"""
        seawater_density = self._material_data["Seawater"]["density_air"]
        radius_m = (self.specs.od_foc_mm / 2) * 1e-3
        area_m2 = math.pi * radius_m ** 2
        displaced_wt = seawater_density * area_m2

        single_seawater_wt = self.specs.weight_single_foc_kg_per_m - displaced_wt
        total_air_wt = self.specs.weight_single_foc_kg_per_m * self.specs.num_focs
        total_seawater_wt = single_seawater_wt * self.specs.num_focs

        return {
            "weight_air_total_kg_per_m": round(total_air_wt, 3),
            "weight_seawater_total_kg_per_m": round(total_seawater_wt, 3)
        }

    def calculate_filler_weight(self) -> Dict:
        """Calculate filler material weight"""
        material = self._material_data[self.specs.filler_material]

        # Area calculations
        a_core = math.pi * ((self.specs.od_power_core_mm / 2) ** 2)
        a_foc = math.pi * ((self.specs.od_foc_mm / 2) ** 2) * self.specs.num_focs
        a_assembled = math.pi * ((self.specs.assembled_od_mm / 2) ** 2)
        a_interstice = ((self.specs.od_power_core_mm / 2) ** 2) * (math.sqrt(3) - math.pi / 2)

        max_fill_area = a_assembled - 3 * a_core - a_foc - a_interstice
        filler_area = self.specs.filler_coverage_percent * max_fill_area

        air_wt = filler_area * material["density_air"] * 1e-6
        seawater_wt = filler_area * material["density_seawater"] * 1e-6

        return {
            "area_filler_mm2": round(filler_area, 3),
            "weight_air_kg_per_m": round(air_wt, 3),
            "weight_seawater_kg_per_m": round(seawater_wt, 3)
        }

    def calculate_layup_assembly(self, core_results: Dict, foc_results: Dict,
                                 filler_results: Dict, cabling_factor: float) -> Dict:
        """Calculate laid-up cable assembly weight"""
        air_wt = cabling_factor * (
                3 * core_results["total_weight_kg_per_m"] +
                foc_results["weight_air_total_kg_per_m"] +
                filler_results["weight_air_kg_per_m"]
        )

        seawater_wt = cabling_factor * (
                3 * core_results["weight_in_seawater_kg_per_m"] +
                foc_results["weight_seawater_total_kg_per_m"] +
                filler_results["weight_seawater_kg_per_m"]
        )

        return {
            "total_weight_air_kg_per_m": round(air_wt, 3),
            "total_weight_seawater_kg_per_m": round(seawater_wt, 3)
        }

    def calculate_bedding_layer(self) -> Dict:
        """Calculate inner bedding layer weight"""
        material = "Polypropylene" if self.specs.design_type == "ROVED" else "HDPE"
        fill_factor = 0.7 if self.specs.design_type == "ROVED" else 1.0

        mat_data = self._material_data[material]
        inner_d = self.specs.bedding_od_mm - 2 * self.specs.bedding_thickness_mm
        area = math.pi * ((self.specs.bedding_od_mm / 2) ** 2 - (inner_d / 2) ** 2)
        volume = fill_factor * area

        air_wt = volume * mat_data["density_air"] * 1e-6
        seawater_wt = volume * mat_data["density_seawater"] * 1e-6

        return {
            "volume_mm3_per_m": round(volume, 3),
            "weight_air_kg_per_m": round(air_wt, 3),
            "weight_seawater_kg_per_m": round(seawater_wt, 3)
        }

    def calculate_armor_layer(self) -> Dict:
        """Calculate armor layer weight with lay correction"""
        # Material selection
        steel_key = f"Stainless Steel Wires (SS)" if self.specs.steel_type == "SS" else "Galvanised Steel Wires (GS)"
        steel_data = self._material_data[steel_key]
        pe_data = self._material_data["MDPE"]

        # Wire areas
        wire_area = math.pi * (self.specs.wire_diameter_mm / 2) ** 2
        steel_area = self.specs.num_steel_wires * wire_area
        pe_area = self.specs.num_pe_wires * wire_area

        # Straight weights
        steel_air_straight = steel_area * steel_data["density_air"] * 1e-6
        steel_seawater_straight = steel_area * steel_data["density_seawater"] * 1e-6
        pe_air_straight = pe_area * pe_data["density_air"] * 1e-6
        pe_seawater_straight = pe_area * pe_data["density_seawater"] * 1e-6

        # Lay correction
        radius_center = (self.specs.bedding_od_mm + self.specs.wire_diameter_mm) / 2
        circumf = 2 * math.pi * radius_center
        helix_len = math.sqrt(self.specs.armor_lay_length_mm ** 2 + circumf ** 2)
        cabling_factor = helix_len / self.specs.armor_lay_length_mm

        # Final weights
        steel_air = cabling_factor * steel_air_straight
        steel_seawater = cabling_factor * steel_seawater_straight
        pe_air = cabling_factor * pe_air_straight
        pe_seawater = cabling_factor * pe_seawater_straight

        # Fill factor
        total_wire_dia = self.specs.wire_diameter_mm * (self.specs.num_steel_wires + self.specs.num_pe_wires)
        fill_factor = (total_wire_dia / circumf) * 100

        return {
            "total_steel_area_mm2": round(steel_area, 3),
            "cabling_factor": round(cabling_factor, 3),
            "fill_factor_percent": round(fill_factor, 3),
            "weight_steel_air_kg_per_m": round(steel_air, 3),
            "weight_steel_seawater_kg_per_m": round(steel_seawater, 3),
            "weight_pe_air_kg_per_m": round(pe_air, 3),
            "weight_pe_seawater_kg_per_m": round(pe_seawater, 3),
            "total_weight_air_kg_per_m": round(steel_air + pe_air, 3),
            "total_weight_seawater_kg_per_m": round(steel_seawater + pe_seawater, 3)
        }

    def calculate_bitumen_flushing(self, steel_area: float, fill_factor: float = 0.9) -> Dict:
        """Calculate bitumen flushing weight"""
        bitumen_data = self._material_data["Bitumen"]

        outer_d = self.specs.bedding_od_mm + 2 * self.specs.wire_diameter_mm
        annulus_area = math.pi * ((outer_d / 2) ** 2 - (self.specs.bedding_od_mm / 2) ** 2)
        interstice_area = annulus_area - steel_area
        bitumen_volume = fill_factor * interstice_area

        air_wt = bitumen_volume * bitumen_data["density_air"] * 1e-6
        seawater_wt = bitumen_volume * bitumen_data["density_seawater"] * 1e-6

        return {
            "bitumen_volume_mm3_per_m": round(bitumen_volume, 3),
            "bitumen_weight_air_kg_per_m": round(air_wt, 3),
            "bitumen_weight_seawater_kg_per_m": round(seawater_wt, 3)
        }

    def calculate_outer_protection(self) -> Dict:
        """Calculate outer roving/sheath weight"""
        material = "Polypropylene" if self.specs.design_type == "ROVED" else "HDPE"
        fill_factor = 0.5 if self.specs.design_type == "ROVED" else 1.0

        mat_data = self._material_data[material]
        inner_d = self.specs.outer_od_mm - 2 * self.specs.outer_thickness_mm
        area = math.pi * ((self.specs.outer_od_mm / 2) ** 2 - (inner_d / 2) ** 2)
        volume = area * fill_factor

        air_wt = volume * mat_data["density_air"] * 1e-6
        seawater_wt = volume * mat_data["density_seawater"] * 1e-6

        return {
            "fill_factor": fill_factor,
            "weight_air_kg_per_m": round(air_wt, 3),
            "weight_seawater_kg_per_m": round(seawater_wt, 3)
        }

    def calculate_total_weight(self) -> Dict:
        """Calculate complete cable weight breakdown"""
        # Calculate all components
        core_results = self.calculate_power_core()
        foc_results = self.calculate_foc_weight()
        filler_results = self.calculate_filler_weight()

        layup_results = self.calculate_layup_assembly(
            core_results, foc_results, filler_results,
            core_results["layup_geometry"]["cabling_factor"]
        )

        bedding_results = self.calculate_bedding_layer()
        armor_results = self.calculate_armor_layer()
        bitumen_results = self.calculate_bitumen_flushing(armor_results["total_steel_area_mm2"])
        outer_results = self.calculate_outer_protection()

        # Total weights
        total_air = (layup_results["total_weight_air_kg_per_m"] +
                     bedding_results["weight_air_kg_per_m"] +
                     armor_results["total_weight_air_kg_per_m"] +
                     bitumen_results["bitumen_weight_air_kg_per_m"] +
                     outer_results["weight_air_kg_per_m"])

        total_seawater = (layup_results["total_weight_seawater_kg_per_m"] +
                          bedding_results["weight_seawater_kg_per_m"] +
                          armor_results["total_weight_seawater_kg_per_m"] +
                          bitumen_results["bitumen_weight_seawater_kg_per_m"] +
                          outer_results["weight_seawater_kg_per_m"])

        # Store all results
        self._results = {
            "power_core": core_results,
            "foc": foc_results,
            "filler": filler_results,
            "layup_assembly": layup_results,
            "bedding": bedding_results,
            "armor": armor_results,
            "bitumen": bitumen_results,
            "outer_protection": outer_results,
            "total_weight_air_kg_per_m": round(total_air, 3),
            "total_weight_seawater_kg_per_m": round(total_seawater, 3)
        }

        return self._results


def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="Submarine Cable Weight Calculator",
        page_icon="⚓",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("⚓ Submarine Cable Weight Calculator")
    st.markdown("Calculate the weight of submarine power cables in air and seawater with detailed component breakdown.")

    # Sidebar for inputs
    st.sidebar.header("📋 Cable Specifications")

    # Initialize session state for specs
    if 'specs' not in st.session_state:
        st.session_state.specs = CableSpecs()

    with st.sidebar:
        st.subheader("🔌 Conductor Specifications")
        conductor_size = st.number_input(
            "Conductor Size (mm²)",
            min_value=0.0,
            value=float(st.session_state.specs.conductor_size_mm2),
            step=50.0
        )
        conductor_material = st.selectbox(
            "Conductor Material",
            ["Al", "Cu"],
            index=0 if st.session_state.specs.conductor_material == "Al" else 1
        )

        st.subheader("📏 Layer Dimensions (mm)")
        inner_semicon_od = st.number_input(
            "Inner Semicon OD",
            min_value=0.0,
            value=float(st.session_state.specs.inner_semicon_od_mm),
            step=1.0
        )
        inner_semicon_thickness = st.number_input(
            "Inner Semicon Thickness",
            min_value=0.0,
            value=float(st.session_state.specs.inner_semicon_thickness_mm),
            step=0.1
        )
        xlpe_od = st.number_input(
            "XLPE OD",
            min_value=0.0,
            value=float(st.session_state.specs.xlpe_od_mm),
            step=1.0
        )
        outer_semicon_od = st.number_input(
            "Outer Semicon OD",
            min_value=0.0,
            value=float(st.session_state.specs.outer_semicon_od_mm),
            step=1.0
        )
        water_block_tape_thickness = st.number_input(
            "Water Block Tape Thickness",
            min_value=0.0,
            value=float(st.session_state.specs.water_block_tape_thickness_mm),
            step=0.1
        )
        water_block_tape_od = st.number_input(
            "Water Block Tape OD",
            min_value=0.0,
            value=float(st.session_state.specs.water_block_tape_od_mm),
            step=1.0
        )
        copper_screen_size = st.number_input(
            "Copper Screen Size (mm²)",
            min_value=0.0,
            value=float(st.session_state.specs.copper_screen_size_mm2),
            step=10.0
        )
        lead_sheath_thickness = st.number_input(
            "Lead Sheath Thickness",
            min_value=0.0,
            value=float(st.session_state.specs.lead_sheath_thickness_mm),
            step=0.1
        )
        lead_sheath_od = st.number_input(
            "Lead Sheath OD",
            min_value=0.0,
            value=float(st.session_state.specs.lead_sheath_od_mm),
            step=1.0
        )
        pe_sheath_thickness = st.number_input(
            "PE Sheath Thickness",
            min_value=0.0,
            value=float(st.session_state.specs.pe_sheath_thickness_mm),
            step=0.1
        )
        od_power_core = st.number_input(
            "Power Core OD",
            min_value=0.0,
            value=float(st.session_state.specs.od_power_core_mm),
            step=1.0
        )

        st.subheader("🔧 Cable Assembly")
        lay_length = st.number_input(
            "Lay Length (mm)",
            min_value=0.0,
            value=float(st.session_state.specs.lay_length_mm),
            step=100.0
        )

        st.subheader("🌐 FOC Specifications")
        od_foc = st.number_input(
            "FOC OD (mm)",
            min_value=0.0,
            value=float(st.session_state.specs.od_foc_mm),
            step=1.0
        )
        weight_single_foc = st.number_input(
            "Single FOC Weight (kg/m)",
            min_value=0.0,
            value=float(st.session_state.specs.weight_single_foc_kg_per_m),
            step=0.01
        )
        num_focs = st.number_input(
            "Number of FOCs",
            min_value=0,
            value=int(st.session_state.specs.num_focs),
            step=1
        )

        st.subheader("🔩 Assembly & Filler")
        assembled_od = st.number_input(
            "Assembled OD (mm)",
            min_value=0.0,
            value=float(st.session_state.specs.assembled_od_mm),
            step=1.0
        )
        filler_coverage = st.slider(
            "Filler Coverage (%)",
            min_value=0.0,
            max_value=1.0,
            value=float(st.session_state.specs.filler_coverage_percent),
            step=0.05
        )
        filler_material = st.selectbox(
            "Filler Material",
            ["Filler_PVC", "Filler_MDPE", "Filler_SC PE"],
            index=0
        )

        st.subheader("🛡️ Bedding & Armor")
        bedding_od = st.number_input(
            "Bedding OD (mm)",
            min_value=0.0,
            value=float(st.session_state.specs.bedding_od_mm),
            step=1.0
        )
        bedding_thickness = st.number_input(
            "Bedding Thickness (mm)",
            min_value=0.0,
            value=float(st.session_state.specs.bedding_thickness_mm),
            step=0.1
        )
        num_steel_wires = st.number_input(
            "Number of Steel Wires",
            min_value=0,
            value=int(st.session_state.specs.num_steel_wires),
            step=1
        )
        num_pe_wires = st.number_input(
            "Number of PE Wires",
            min_value=0,
            value=int(st.session_state.specs.num_pe_wires),
            step=1
        )
        wire_diameter = st.number_input(
            "Wire Diameter (mm)",
            min_value=0.0,
            value=float(st.session_state.specs.wire_diameter_mm),
            step=0.1
        )
        armor_lay_length = st.number_input(
            "Armor Lay Length (mm)",
            min_value=0.0,
            value=float(st.session_state.specs.armor_lay_length_mm),
            step=100.0
        )
        steel_type = st.selectbox(
            "Steel Type",
            ["GS", "SS"],
            index=0 if st.session_state.specs.steel_type == "GS" else 1
        )

        st.subheader("🔒 Outer Protection")
        outer_od = st.number_input(
            "Outer OD (mm)",
            min_value=0.0,
            value=float(st.session_state.specs.outer_od_mm),
            step=1.0
        )
        outer_thickness = st.number_input(
            "Outer Thickness (mm)",
            min_value=0.0,
            value=float(st.session_state.specs.outer_thickness_mm),
            step=0.1
        )
        design_type = st.selectbox(
            "Design Type",
            ["ROVED", "SHEATHED"],
            index=0 if st.session_state.specs.design_type == "ROVED" else 1
        )

    # Update specs with user inputs
    specs = CableSpecs(
        conductor_size_mm2=conductor_size,
        conductor_material=conductor_material,
        inner_semicon_od_mm=inner_semicon_od,
        inner_semicon_thickness_mm=inner_semicon_thickness,
        xlpe_od_mm=xlpe_od,
        outer_semicon_od_mm=outer_semicon_od,
        water_block_tape_thickness_mm=water_block_tape_thickness,
        water_block_tape_od_mm=water_block_tape_od,
        copper_screen_size_mm2=copper_screen_size,
        lead_sheath_thickness_mm=lead_sheath_thickness,
        lead_sheath_od_mm=lead_sheath_od,
        pe_sheath_thickness_mm=pe_sheath_thickness,
        od_power_core_mm=od_power_core,
        lay_length_mm=lay_length,
        od_foc_mm=od_foc,
        weight_single_foc_kg_per_m=weight_single_foc,
        num_focs=num_focs,
        assembled_od_mm=assembled_od,
        filler_coverage_percent=filler_coverage,
        filler_material=filler_material,
        bedding_od_mm=bedding_od,
        bedding_thickness_mm=bedding_thickness,
        num_steel_wires=num_steel_wires,
        num_pe_wires=num_pe_wires,
        wire_diameter_mm=wire_diameter,
        armor_lay_length_mm=armor_lay_length,
        steel_type=steel_type,
        outer_od_mm=outer_od,
        outer_thickness_mm=outer_thickness,
        design_type=design_type
    )

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("📊 Calculation Results")

        if st.button("🔄 Calculate Cable Weight", type="primary"):
            try:
                calculator = CableWeightCalculator(specs, show_breakdown=True)
                results = calculator.calculate_total_weight()

                # Store results in session state
                st.session_state.results = results
                st.session_state.calculator = calculator

                st.success("✅ Calculation completed successfully!")

            except Exception as e:
                st.error(f"❌ Error in calculation: {str(e)}")

    with col2:
        st.header("💾 Presets")

        if st.button("📁 Load Default Values"):
            st.session_state.specs = CableSpecs()
            st.rerun()

        if st.button("🗑️ Clear All Fields"):
            for key in st.session_state.keys():
                if key.startswith('specs'):
                    del st.session_state[key]
            st.rerun()

    # Display results if available
    if hasattr(st.session_state, 'results') and st.session_state.results:
        results = st.session_state.results

        st.header("🎯 Final Results")

        # Main results cards
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Total Weight in Air",
                f"{results['total_weight_air_kg_per_m']:.3f} kg/m",
                delta=None
            )

        with col2:
            st.metric(
                "Total Weight in Seawater",
                f"{results['total_weight_seawater_kg_per_m']:.3f} kg/m",
                delta=None
            )

        with col3:
            buoyancy = results['total_weight_air_kg_per_m'] - results['total_weight_seawater_kg_per_m']
            st.metric(
                "Buoyancy",
                f"{buoyancy:.3f} kg/m",
                delta=None
            )

        # Technical details
        st.subheader("🔧 Technical Details")
        layup = results['power_core']['layup_geometry']

        tech_col1, tech_col2, tech_col3, tech_col4 = st.columns(4)

        with tech_col1:
            st.metric("Cabling Factor", f"{layup['cabling_factor']:.5f}")

        with tech_col2:
            st.metric("Layup Angle", f"{layup['layup_angle_deg']:.2f}°")

        with tech_col3:
            st.metric("Core Density", f"{results['power_core']['overall_density_kg_per_m3']:.0f} kg/m³")

        with tech_col4:
            st.metric("Armor Fill Factor", f"{results['armor']['fill_factor_percent']:.1f}%")

        # Detailed breakdown
        st.subheader("📋 Detailed Weight Breakdown")

        breakdown_data = {
            "Component": [
                "Power Core (single)",
                "FOC Total",
                "Filler",
                "Laid-up Assembly",
                "Bedding Layer",
                "Armor Layer",
                "Bitumen Flushing",
                "Outer Protection",
                "TOTAL"
            ],
            "Weight in Air (kg/m)": [
                results['power_core']['total_weight_kg_per_m'],
                results['foc']['weight_air_total_kg_per_m'],
                results['filler']['weight_air_kg_per_m'],
                results['layup_assembly']['total_weight_air_kg_per_m'],
                results['bedding']['weight_air_kg_per_m'],
                results['armor']['total_weight_air_kg_per_m'],
                results['bitumen']['bitumen_weight_air_kg_per_m'],
                results['outer_protection']['weight_air_kg_per_m'],
                results['total_weight_air_kg_per_m']
            ],
            "Weight in Seawater (kg/m)": [
                results['power_core']['weight_in_seawater_kg_per_m'],
                results['foc']['weight_seawater_total_kg_per_m'],
                results['filler']['weight_seawater_kg_per_m'],
                results['layup_assembly']['total_weight_seawater_kg_per_m'],
                results['bedding']['weight_seawater_kg_per_m'],
                results['armor']['total_weight_seawater_kg_per_m'],
                results['bitumen']['bitumen_weight_seawater_kg_per_m'],
                results['outer_protection']['weight_seawater_kg_per_m'],
                results['total_weight_seawater_kg_per_m']
            ]
        }

        df_breakdown = pd.DataFrame(breakdown_data)

        # Style the dataframe
        def highlight_total(row):
            if row.name == len(df_breakdown) - 1:  # Last row (TOTAL)
                return ['background-color: #e6f3ff; font-weight: bold'] * len(row)
            return [''] * len(row)

        styled_df = df_breakdown.style.apply(highlight_total, axis=1).format({
            'Weight in Air (kg/m)': '{:.3f}',
            'Weight in Seawater (kg/m)': '{:.3f}'
        })

        st.dataframe(styled_df, use_container_width=True)

        # Power core component breakdown
        st.subheader("🔌 Power Core Component Breakdown")

        core_components = results['power_core']['component_weights']
        core_data = {
            "Component": [
                "Conductor",
                "Waterblock",
                "Inner Semicon",
                "XLPE",
                "Outer Semicon",
                "Water Tape",
                "Copper Screen",
                "Lead Sheath",
                "PE Sheath"
            ],
            "Weight (kg/m)": [
                core_components['conductor'],
                core_components['waterblock'],
                core_components['inner_semicon'],
                core_components['xlpe'],
                core_components['outer_semicon'],
                core_components['water_tape'],
                core_components['copper_screen'],
                core_components['lead_sheath'],
                core_components['pe_sheath']
            ]
        }

        df_core = pd.DataFrame(core_data)
        st.dataframe(df_core.style.format({'Weight (kg/m)': '{:.3f}'}), use_container_width=True)

        # Export functionality
        st.subheader("📤 Export Results")

        col1, col2 = st.columns(2)

        with col1:
            # Prepare export data
            export_data = {
                "Cable Specifications": specs.__dict__,
                "Results Summary": {
                    "Total Weight Air (kg/m)": results['total_weight_air_kg_per_m'],
                    "Total Weight Seawater (kg/m)": results['total_weight_seawater_kg_per_m'],
                    "Buoyancy (kg/m)": buoyancy,
                    "Cabling Factor": layup['cabling_factor'],
                    "Layup Angle (deg)": layup['layup_angle_deg'],
                    "Core Density (kg/m3)": results['power_core']['overall_density_kg_per_m3'],
                    "Armor Fill Factor (%)": results['armor']['fill_factor_percent']
                },
                "Detailed Breakdown": breakdown_data
            }

            if st.download_button(
                    label="📊 Download Results (CSV)",
                    data=df_breakdown.to_csv(index=False),
                    file_name="cable_weight_results.csv",
                    mime="text/csv"
            ):
                st.success("Results exported successfully!")

        with col2:
            if st.button("🖨️ Print Summary"):
                if hasattr(st.session_state, 'calculator'):
                    with st.expander("Print Summary", expanded=True):
                        st.text(f"""
SUBMARINE CABLE WEIGHT CALCULATION SUMMARY
=========================================

Cable Specifications:
- Conductor: {specs.conductor_size_mm2} mm² {specs.conductor_material}
- Power Core OD: {specs.od_power_core_mm} mm
- Final Cable OD: {specs.outer_od_mm} mm
- Design Type: {specs.design_type}

Final Results:
- Total Weight in Air:     {results['total_weight_air_kg_per_m']:8.3f} kg/m
- Total Weight in Seawater: {results['total_weight_seawater_kg_per_m']:8.3f} kg/m
- Buoyancy:               {buoyancy:8.3f} kg/m

Technical Details:
- Cabling Factor:         {layup['cabling_factor']:8.5f}
- Layup Angle:            {layup['layup_angle_deg']:8.2f}°
- Core Density:           {results['power_core']['overall_density_kg_per_m3']:8.0f} kg/m³
- Armor Fill Factor:      {results['armor']['fill_factor_percent']:8.1f}%
                        """)

    # Footer
    st.markdown("---")
    st.markdown(
        "💡 **Tip:** Adjust the cable specifications in the sidebar and click 'Calculate Cable Weight' to see updated results."
    )
    st.markdown(
        "⚠️ **Note:** This calculator is for engineering estimation purposes. Always verify results with detailed engineering analysis from cable supplier specsheet."
    )


if __name__ == "__main__":
    main()