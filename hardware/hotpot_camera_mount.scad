// Hotpot camera mount, v0.1. All dimensions in mm.
// OpenSCAD: openscad -o stl/clamp_body.stl -D 'part="clamp_body"' hotpot_camera_mount.scad
// part = clamp_body | moving_pad | link | pivot | camera_head | print_layout | assembly
// STL exports are already oriented for printing; the assembly is illustrative.

part = "assembly";
$fn = 64;

// Practical defaults for a light webcam and a short arm. Tune these after fit tests.
clamp_width = 32;
clamp_back = 18;
clamp_reach = 48;
jaw_thickness = 14;
clamp_gap = 58;       // Gross space between jaws, before moving pad/contact pads.
pad_thickness = 9;
screw_x = 30;
m8_clearance = 8.8;
m8_nut_flats = 13.5;
m8_nut_depth = 7.2;
m6_clearance = 6.6;
link_centres = 85;
link_width = 26;
link_thickness = 8;
head_length = 72;
head_width = 52;
head_thickness = 6;
slot_width = 6.8;

// A triangle above a round hole limits an unsupported horizontal-hole roof.
module teardrop_2d(d) {
    union() {
        circle(d=d);
        polygon([[-d/2/sqrt(2), d/2/sqrt(2)], [0, d/2*sqrt(2)],
                 [d/2/sqrt(2), d/2/sqrt(2)]]);
    }
}

// Capsule in XY, for arm outline and camera screw slots.
module capsule_2d(length, width) {
    hull() {
        translate([-length/2 + width/2,0]) circle(d=width);
        translate([ length/2 - width/2,0]) circle(d=width);
    }
}

// Clamp in assembly coordinates: table in X/Y, adjustment screw in Z.
module clamp_native() {
    pivot_z = jaw_thickness*2 + clamp_gap + 13;
    difference() {
        union() {
            translate([-clamp_back,-clamp_width/2,0])
                cube([clamp_back,clamp_width,jaw_thickness*2+clamp_gap]);
            translate([0,-clamp_width/2,0])
                cube([clamp_reach,clamp_width,jaw_thickness]);
            translate([0,-clamp_width/2,jaw_thickness+clamp_gap])
                cube([clamp_reach,clamp_width,jaw_thickness]);
            // Round mounting boss; solid across the clamp's full width.
            translate([-clamp_back/2,0,pivot_z]) rotate([90,0,0])
                cylinder(h=clamp_width,d=30,center=true);
            // The boss overlaps the back and upper jaw, not a floating appendage.
            translate([-clamp_back,-clamp_width/2,jaw_thickness+clamp_gap])
                cube([clamp_back,clamp_width,25]);
        }
        // M8 screw tunnel. Apex faces up in the supplied side-print orientation.
        translate([screw_x,0,-1]) linear_extrude(jaw_thickness+2)
            teardrop_2d(m8_clearance);
        // Nut enters from the TOP of the lower jaw and seats against downward force.
        translate([screw_x,0,jaw_thickness-m8_nut_depth])
            cylinder(h=m8_nut_depth+1,d=m8_nut_flats/cos(30),$fn=6);
        translate([-clamp_back/2,0,pivot_z]) rotate([90,0,0])
            cylinder(h=clamp_width+2,d=m6_clearance,center=true);
    }
}

module clamp_body() {
    // Lie on the wide side. No long unsupported bridge across the clamp throat.
    translate([0,jaw_thickness*2+clamp_gap+28,clamp_width/2])
        rotate([90,0,0]) clamp_native();
}

module moving_pad() {
    // Contact face DOWN; blind screw-tip socket opens upward for printing.
    difference() {
        linear_extrude(pad_thickness)
            hull() {
                for(x=[-13,13],y=[-10,10]) translate([x,y]) circle(r=3);
            }
        translate([0,0,pad_thickness-4.5])
            cylinder(d=9,h=5.5);
    }
}

module link() {
    // Two M6 pivots, centre spacing 85 mm; print flat.
    difference() {
        linear_extrude(link_thickness)
            capsule_2d(link_centres+link_width,link_width);
        for(x=[-link_centres/2,link_centres/2])
            translate([x,0,-1]) cylinder(d=m6_clearance,h=link_thickness+2);
    }
}

module pivot() {
    // Right-angle adapter gives a second adjustment axis at the camera head.
    // Print on the square base; the small teardrop holes are horizontal.
    adapter_width=32;
    adapter_height=32;
    wall=8;
    hole_offset=20;
    difference() {
        union() {
            cube([adapter_width,wall,adapter_height]);
            cube([wall,adapter_width,adapter_height]);
            cube([adapter_width,adapter_width,5]);
            // Inner-corner gusset stays clear of the bolt/washer locations.
            linear_extrude(17) polygon([[wall,wall],[wall+8,wall],[wall,wall+8]]);
        }
        translate([hole_offset,wall+1,20]) rotate([90,0,0])
            linear_extrude(wall+2) teardrop_2d(m6_clearance);
        translate([-1,hole_offset,20]) rotate([90,0,90])
            linear_extrude(wall+2) teardrop_2d(m6_clearance);
    }
}

module camera_head() {
    // Flat face sits on bed. After assembly this face becomes the camera's top.
    // A camera-specific adapter may be bolted to these slots later.
    difference() {
        union() {
            translate([-head_length/2,-head_width/2,0])
                cube([head_length,head_width,head_thickness]);
            // Upright pitch/roll ear, joined across its full base.
            translate([-24,-4,head_thickness]) cube([24,8,12]);
            translate([-12,0,head_thickness+12]) rotate([90,0,0])
                cylinder(h=8,d=24,center=true);
        }
        for(y=[-15,0,15])
            translate([7,y,-1]) linear_extrude(head_thickness+2)
                capsule_2d(42,slot_width);
        translate([-12,5,head_thickness+12]) rotate([90,0,0])
            linear_extrude(10) teardrop_2d(m6_clearance);
    }
}

module print_layout() {
    translate([25,0,0]) clamp_body();
    translate([105,20,0]) moving_pad();
    translate([115,70,0]) link();
    translate([170,5,0]) pivot();
    translate([215,85,0]) camera_head();
}

module illustrative_assembly() {
    // Two links nearly upright; one link is also usable for a shorter fixture.
    pivot_z = jaw_thickness*2+clamp_gap+13;
    start = [-clamp_back/2,clamp_width/2,pivot_z];
    theta = 65;
    middle = [start[0]+link_centres*cos(theta),start[1]+link_thickness,
              start[2]+link_centres*sin(theta)];
    endpoint = [middle[0]+link_centres*cos(theta),middle[1]+link_thickness,
                middle[2]+link_centres*sin(theta)];
    color("#f0b84b") clamp_native();
    // Moving pad: contact face above the socket.
    color("#f0b84b") translate([screw_x,0,42]) rotate([180,0,0]) moving_pad();
    color("#778899") translate([screw_x,0,-24]) cylinder(d=8,h=61);
    color("#5a9ba5") translate(start) rotate([0,-theta,0])
        rotate([90,0,0]) translate([link_centres/2,0,-link_thickness]) link();
    color("#5a9ba5") translate(middle) rotate([0,-theta,0])
        rotate([90,0,0]) translate([link_centres/2,0,-link_thickness]) link();
    color("#6bb7c1") translate([endpoint[0]-20,endpoint[1],endpoint[2]-20]) pivot();
    // Head's ear normal X; slot face points upwards.
    color("#b6d5d7") translate([endpoint[0]-20-4,endpoint[1]+20,
                                 endpoint[2]+18]) rotate([180,0,90])
        translate([12,0,0]) camera_head();
    // Table edge reference, shown only in assembly.
    color([0.45,0.38,0.30,0.32]) translate([0,-70,42]) cube([100,140,28]);
}

if (part=="clamp_body") clamp_body();
else if (part=="moving_pad") moving_pad();
else if (part=="link") link();
else if (part=="pivot") pivot();
else if (part=="camera_head") camera_head();
else if (part=="print_layout") print_layout();
else if (part=="assembly") illustrative_assembly();
else assert(false,str("Unknown part: ",part));
