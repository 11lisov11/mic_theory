"""Export standalone 24/18 rotor concepts with FreeCAD's bundled Python (mm)."""
import argparse
import math
from pathlib import Path

from build_air56 import App, Part, ring, sector


def build(out, name, modulation, skew_deg, insert=False, tag=False):
    doc = App.newDocument(name)
    def add(label, shape):
        if shape.isNull() or not shape.isValid():
            raise ValueError(label)
        item = doc.addObject("Part::Feature", label)
        item.Shape = shape
        return item

    bars = []
    for i in range(18):
        angle = (i+.5)*math.tau/18
        width = math.pi/18*.39*(1+modulation*math.cos(2*angle))
        wires = [sector(20., 23.5, angle+math.radians(skew_deg)*(j/4-.5), width, -27+54*j/4)
                 for j in range(5)]
        bars.append(Part.makeLoft(wires, True))
    steel = ring(24., 8., 54., -27).cut(Part.makeCompound(bars))
    if insert:
        patch = Part.makeCylinder(24.1,54,App.Vector(0,0,-27),App.Vector(0,0,1),20)
        patch.rotate(App.Vector(),App.Vector(0,0,1),-10)
        patch = patch.cut(Part.makeCylinder(23.5,54,App.Vector(0,0,-27)))
        add("Insert", steel.common(patch))
        steel = steel.cut(patch)
    if tag:
        # A packaging envelope on the rotor end, outside the active air gap.
        add("LC_package_envelope", Part.makeBox(4, 4, .6, App.Vector(12,-2,27)))
    add("LaminatedCoreEnvelope", steel)
    for i, bar in enumerate(bars):
        add(f"Bar{i:02}", bar)
    for i, z in enumerate((-29.4, 27)):
        add(f"EndRing{i}", ring(23.5,19.,2.4,z))
    doc.recompute()
    doc.saveAs(str(out/f"{name}.FCStd"))
    path = out/f"{name}.step"
    Part.export(doc.Objects, str(path))
    expected = sum(len(obj.Shape.Solids) for obj in doc.Objects)
    if len(Part.read(str(path)).Solids) != expected:
        raise ValueError("STEP solid count")
    App.closeDocument(doc.Name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    out = parser.parse_args().out
    out.mkdir(parents=True, exist_ok=True)
    for name, width, skew, insert, tag in (("Bars60", .6, 0, False, False), ("Bars60Skew", .6, 20, False, False),
                                          ("Insert", 0, 0, True, False), ("LC", 0, 0, False, True)):
        build(out,name,width,skew,insert,tag)
