// Low-slip electrokinetic cage cross-check. Geometry uses metres.
DefineConstant[ Quadrature = 0, Sigma = 35000000 ];
Group { Cage = Region[1]; Gauge = Region[2]; }
Function {
  applied[] = Vector[0,0, (Fabs[Z[]]<0.027) *
                 (Quadrature ? Y[] : X[])/Sqrt[X[]^2+Y[]^2]];
}
Constraint { { Name Ground; Type Assign; Case { { Region Gauge; Value 0.; } } } }
Jacobian { { Name J; Case { { Region All; Jacobian Vol; } } } }
Integration { { Name I; Case { { Type Gauss; Case { { GeoElement Tetrahedron; NumberOfPoints 4; } } } } } }
FunctionSpace { { Name H1; Type Form0;
  BasisFunction { { Name sn; NameOfCoef vn; Function BF_Node; Support Cage; Entity NodesOf[All]; } }
  Constraint { { NameOfCoef vn; EntityType NodesOf; NameOfConstraint Ground; } }
} }
Formulation { { Name Current; Type FemEquation;
  Quantity { { Name v; Type Local; NameOfSpace H1; } }
  Equation {
    Integral { [ Dof{d v}, {d v} ]; In Cage; Jacobian J; Integration I; }
    Integral { [ -applied[], {d v} ]; In Cage; Jacobian J; Integration I; }
  }
} }
Resolution { { Name SolveCage; System { { Name A; NameOfFormulation Current; } }
  Operation { Generate[A]; Solve[A]; SaveSolution[A]; } } }
PostProcessing { { Name Power; NameOfFormulation Current; Quantity {
  { Name Loss; Value { Integral { [ 0.5*Sigma*SquNorm[applied[]-{d v}] ]; In Cage; Jacobian J; Integration I; } } }
  { Name Input; Value { Integral { [ 0.5*Sigma*(applied[]*(applied[]-{d v})) ]; In Cage; Jacobian J; Integration I; } } }
} } }
PostOperation { { Name Balance; NameOfPostProcessing Power; Operation {
  Print[Loss[Cage],OnGlobal,Format Table,File "loss.txt"];
  Print[Input[Cage],OnGlobal,Format Table,File "input.txt"];
} } }
