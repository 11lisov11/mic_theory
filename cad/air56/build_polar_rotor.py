"""Place the two remanent sectors in existing AIR56 18-bar candidate CAD."""

import argparse
from pathlib import Path
import math
import json
from build_air56 import App, Part


def pocket_wire(theta, z):
    def point(r, a):
        return App.Vector(r * math.cos(a), r * math.sin(a), z)

    a, b = theta - math.radians(8), theta + math.radians(8)
    return Part.Wire(
        [
            Part.Arc(point(24.2, a), point(24.2, theta), point(24.2, b)).toShape(),
            Part.makeLine(point(24.2, b), point(23.52, b)),
            Part.Arc(point(23.52, b), point(23.52, theta), point(23.52, a)).toShape(),
            Part.makeLine(point(23.52, a), point(24.2, a)),
        ]
    )


def export(source, out):
    out.mkdir(parents=True, exist_ok=True)
    checks = []
    for name, skew in (("Bars60", 0.0), ("Bars60Skew", 20.0)):
        doc = App.openDocument(str(source / (name + ".FCStd")))
        core = doc.getObject("LaminatedCoreEnvelope")
        volume = core.Shape.Volume
        added = 0.0
        for i, direction in enumerate(("NorthOut", "SouthOut")):
            wires = [
                pocket_wire(i * math.pi + math.radians(skew) * (j / 8 - 0.5), -27 + 54 * j / 8)
                for j in range(9)
            ]
            cavity = (
                Part.makeLoft(wires, True) if skew else Part.Face(wires[0]).extrude(App.Vector(0, 0, 54))
            )
            insert = doc.addObject("Part::Feature", direction)
            insert.addProperty("App::PropertyString", "Magnetization")
            insert.Magnetization = "0.4 T parallel to rotor +X in the local section"
            insert.Shape = core.Shape.common(cavity)
            core.Shape = core.Shape.cut(cavity)
            added += insert.Shape.Volume
            if len(insert.Shape.Solids) != 1 or not insert.Shape.isValid():
                raise ValueError("insert geometry")
        if abs(volume - core.Shape.Volume - added) > 1e-4 * added:
            raise ValueError("volume conservation")
        doc.recompute()
        stem = "Polar18Skew" if skew else "Polar18"
        doc.saveAs(str(out / (stem + ".FCStd")))
        path = out / (stem + ".step")
        Part.export(doc.Objects, str(path))
        solids = Part.read(str(path)).Solids
        if len(solids) != 23 or not all(s.isValid() for s in solids):
            raise ValueError("STEP round trip")
        checks.append(dict(name=stem, solids=len(solids), insert_volume_mm3=added, skew_deg=skew))
        App.closeDocument(doc.Name)
    (out / "checks.json").write_text(json.dumps(checks, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    export(a.source.resolve(), a.out.resolve())
