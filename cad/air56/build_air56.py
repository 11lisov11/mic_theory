"""Build the reconstructed AIR56 candidate with FreeCAD's bundled Python."""

import argparse
import json
import math
from pathlib import Path

import FreeCAD as App
import Import
import Part


ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / "research/air56_interlaminar_marker/geometry.json"
MATERIALS = {
    "electrical steel": "#657a90", "shaft steel": "#bac6cf", "copper": "#ca753c",
    "cage aluminium": "#b1b8c2", "slot insulation": "#e0c381", "housing aluminium": "#2f596c",
    "bearing steel": "#a2a7af", "fan polymer": "#4d7b82", "contact equivalent": "#e84942",
}


def cyl(radius, length, z):
    return Part.makeCylinder(radius, length, App.Vector(0, 0, z))


def ring(outer, inner, length, z):
    return cyl(outer, length, z).cut(cyl(inner, length, z))


def sector(inner, outer, center, halfwidth, z):
    angles = [center+halfwidth*i/4 for i in range(-4, 5)]
    def point(r, a):
        return App.Vector(r*math.cos(a), r*math.sin(a), z)
    vertices = [point(outer, a) for a in angles] + [point(inner, a) for a in reversed(angles)]
    return Part.makePolygon(vertices+[vertices[0]])


def part_of_slot(inner, outer, center, width, z, length):
    return Part.Face(sector(inner, outer, center, width, z)).extrude(App.Vector(0, 0, length))


def assembly(config, name, angle_deg=0, exploded=False, marker=True):
    doc = App.newDocument(name)
    registry = doc.addObject("Spreadsheet::Sheet", "GeometryPrior")
    registry.set("A1", "Dimension (mm)")
    registry.set("B1", "Reconstruction")
    registry.set("C1", "Min")
    registry.set("D1", "Max")
    for row, (key, values) in enumerate(config["dimensions_mm"].items(), 2):
        for col, value in zip("ABCD", (key, values[1], values[0], values[2])):
            registry.set(col+str(row), str(value))
    groups = {}
    for group in ("Housing", "Stator", "Rotor", "Front", "Rear"):
        groups[group] = doc.addObject("App::Part", group)
    groups["Rotor"].Placement = App.Placement(App.Vector(), App.Rotation(App.Vector(0, 0, 1), angle_deg))
    catalog = []

    def feature(group, label, shape, material):
        if shape.isNull() or not shape.isValid() or not shape.Solids:
            raise ValueError(f"Invalid solid: {label}")
        item = groups[group].newObject("Part::Feature", label)
        item.Shape = shape
        item.addProperty("App::PropertyString", "Material", "Reconstruction")
        item.Material = material
        if item.ViewObject is not None:
            rgb = MATERIALS[material].lstrip("#")
            item.ViewObject.ShapeColor = tuple(int(rgb[i:i+2], 16)/255 for i in (0, 2, 4))
        catalog.append((group, item, MATERIALS[material]))
        return item

    d = {key: values[1] for key, values in config["dimensions_mm"].items()}
    rr, gap, sr, shaft, stack = (d[key] for key in ("rotor_radius", "air_gap", "stator_radius",
                                                    "shaft_radius", "stack_length"))
    bore = rr+gap
    z = -stack/2
    ns, nr = config["stator_slots"][1], config["rotor_bars"][1]
    tooth_depth = d["stator_slot_depth"]
    sd = math.pi*config["stator_slot_fraction"]/ns
    rd = math.pi*config["rotor_bar_fraction"]/nr
    bar_inner = rr-d["rotor_bar_depth"]
    bar_outer = rr-0.24
    stator_angles = [i*2*math.pi/ns for i in range(ns)]
    rotor_angles = [i*2*math.pi/nr for i in range(nr)]
    winding_regions = [part_of_slot(bore-0.05, bore+tooth_depth, a, sd, z-0.1, stack+0.2)
                       for a in stator_angles]
    stator = ring(sr, bore, stack, z).cut(Part.makeCompound(winding_regions))
    feature("Stator", "SlottedStatorStack", stator, "electrical steel")
    for i, a in enumerate(stator_angles):
        outline = part_of_slot(bore+0.15, bore+tooth_depth-0.10, a, sd*0.96, z, stack)
        copper = part_of_slot(bore+0.42, bore+tooth_depth-0.44, a, sd*0.82, z-2, stack+4)
        feature("Stator", f"SlotLiner{i:02}", outline.cut(copper), "slot insulation")
        # Double layer: outgoing and incoming regions of a simplified coil.
        copper_low = part_of_slot(bore+0.42, bore+tooth_depth*0.47, a, sd*0.82, z-2, stack+4)
        copper_high = part_of_slot(bore+tooth_depth*0.54, bore+tooth_depth-0.44, a, sd*0.82, z-2, stack+4)
        phase = ("APlus", "CMinus", "BPlus", "AMinus", "CPlus", "BMinus")[i*6//ns]
        feature("Stator", f"Phase{phase}Slot{i:02}Inner", copper_low, "copper")
        feature("Stator", f"Phase{phase}Slot{i:02}Outer", copper_high, "copper")
    for pitch, a in enumerate(stator_angles):
        if pitch % 2:
            continue
        arc_half = math.pi*0.42
        for label, end in (("Drive", z-5.5), ("Fan", -z+3.8)):
            coil_end = part_of_slot(bore+2.1+pitch%3, bore+2.8+pitch%3, a+arc_half, arc_half, end, 1.7)
            feature("Stator", f"{label}EndTurn{pitch:02}", coil_end, "copper")
    skew = math.radians(config["rotor_skew_deg"])

    def rotor_profile(a, level):
        return sector(bar_inner, bar_outer, a+skew*(level-z)/stack, rd, level)

    bars = [Part.makeLoft([rotor_profile(a, z-2.4), rotor_profile(a, -z+2.4)], True)
            for a in rotor_angles]
    bar_cut = Part.makeCompound(bars)
    sheet = d["sheet_thickness"]
    insulation = d["insulation_gap"]
    mid_height = 3*sheet+2*insulation
    pack_length = (stack-mid_height)/2
    for label, start, size in (("RotorPackDrive", z, pack_length),
                               ("RotorPackFan", mid_height/2, pack_length)):
        feature("Rotor", label, ring(rr, shaft, size, start).cut(bar_cut), "electrical steel")
    for n in range(3):
        start = -mid_height/2+n*(sheet+insulation)
        feature("Rotor", f"ExplicitSheet{n+1}", ring(rr, shaft, sheet, start).cut(bar_cut), "electrical steel")
    for i, bar in enumerate(bars):
        feature("Rotor", f"CageBar{i:02}", bar, "cage aluminium")
    for i, level in enumerate((z-2.4, -z)):
        feature("Rotor", f"ShortCircuitRing{i}", ring(rr-0.05, bar_inner-0.8, 2.4, level), "cage aluminium")
    if marker:
        width = d["contact_width"]
        contact_radius = rr-2
        level = -mid_height/2+sheet
        for i, position in enumerate(config["contact_locations_local_mm"]):
            a = position/contact_radius
            bridge = part_of_slot(contact_radius-width/2, contact_radius+width/2, a, width/(2*contact_radius),
                                  level, insulation)
            contact = feature("Rotor", f"InterlaminarBridge{i}", bridge, "contact equivalent")
            contact.addProperty("App::PropertyFloat", "ResistanceOhm", "Reconstruction")
            contact.ResistanceOhm = 1e-4
    feature("Rotor", "Shaft", cyl(shaft, stack+64, z-36), "shaft steel")
    housing_radius = d["housing_radius"]
    feature("Housing", "MotorHousing", ring(housing_radius, sr+0.18, stack+8, z-4), "housing aluminium")
    for i in range(28):
        a = i*2*math.pi/28
        fin = part_of_slot(housing_radius-0.2, housing_radius+3.1, a, 0.018, z-2, stack+4)
        feature("Housing", f"CoolingFin{i:02}", fin, "housing aluminium")
    foot_top_y = -housing_radius+2
    foot_y = -d["mounting_height"]
    for side, xx in (("L", -28), ("R", 17)):
        foot = Part.makeBox(11, foot_top_y-foot_y, stack+8, App.Vector(xx, foot_y, z-4))
        feature("Housing", f"Foot{side}", foot, "housing aluminium")
    for group, zz, orientation in (("Front", z-9, -1), ("Rear", -z+3, 1)):
        flange = ring(housing_radius, shaft+5.3, 6, zz)
        bearing_z = zz+1
        feature(group, group+"Shield", flange, "housing aluminium")
        feature(group, group+"Bearing", ring(shaft+5.25, shaft, 4, bearing_z), "bearing steel")
        for i in range(4):
            a = i*math.pi/2+math.pi/4
            fastener = cyl(1.9, 3, zz-2 if orientation < 0 else zz+6)
            fastener.translate(App.Vector((sr+2)*math.cos(a), (sr+2)*math.sin(a), 0))
            feature(group, f"{group}Fastener{i}", fastener, "bearing steel")
    fan_z = -z+14
    feature("Rear", "FanHub", ring(shaft+6.5, shaft, 4, fan_z), "fan polymer")
    for i in range(9):
        fan = part_of_slot(shaft+6.2, sr-4, i*2*math.pi/9, 0.081, fan_z, 3)
        feature("Rear", f"FanBlade{i}", fan, "fan polymer")
    feature("Rear", "FanCover", ring(housing_radius, housing_radius-2, 17, -z+10), "housing aluminium")
    if exploded:
        groups["Front"].Placement = App.Placement(App.Vector(0, 0, -50), groups["Front"].Placement.Rotation)
        groups["Stator"].Placement = App.Placement(App.Vector(0, 0, -10), groups["Stator"].Placement.Rotation)
        groups["Rotor"].Placement = App.Placement(App.Vector(0, 0, 49), groups["Rotor"].Placement.Rotation)
        groups["Rear"].Placement = App.Placement(App.Vector(0, 0, 108), groups["Rear"].Placement.Rotation)
    doc.recompute()
    return doc, groups, catalog


def export_scene(catalog, groups, path):
    def export_one(group, item, color):
        shape = item.Shape.copy()
        shape.Placement = groups[group].Placement.multiply(shape.Placement)
        points, faces = shape.tessellate(0.55)
        return {"part": item.Name, "group": group, "color": color,
                "vertices": [[p.x, p.y, p.z] for p in points], "faces": faces}
    path.write_text(json.dumps([export_one(*entry) for entry in catalog], separators=(",", ":")), encoding="utf8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", type=Path, default=DEFAULT)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--angle", type=float, default=0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.geometry.read_text(encoding="utf8"))
    results = {}
    for suffix, exploded, marker in (("", False, True), ("_exploded", True, True),
                                     ("_control", False, False)):
        name = "AIR56" + suffix
        doc, groups, catalog = assembly(config, name, args.angle, exploded, marker)
        doc.saveAs(str(args.out/(name+".FCStd")))
        full = args.out/(name+".step")
        Import.export(list(groups.values()), str(full))
        results[name] = {"objects": len(catalog), "step_solids": len(Part.read(str(full)).Solids)}
        if not suffix:
            box = Part.makeBox(250, 100, 250, App.Vector(-125, 0, -125))
            section = []
            for group, item, _color in catalog:
                shape = item.Shape.copy()
                shape.Placement = groups[group].Placement.multiply(shape.Placement)
                clipped = shape.common(box)
                if clipped.Solids:
                    section.append(clipped)
            Part.makeCompound(section).exportStep(str(args.out/"AIR56_section.step"))
            export_scene(catalog, groups, args.out/"cad_scene.json")
        App.closeDocument(doc.Name)
    (args.out/"cad_summary.json").write_text(json.dumps({"freecad": App.Version()[:3], "results": results},
                                                        indent=2), encoding="utf8")
    if any(row["objects"] != row["step_solids"] for row in results.values()):
        raise RuntimeError("STEP readback lost solids")


if __name__ == "__main__":
    main()
