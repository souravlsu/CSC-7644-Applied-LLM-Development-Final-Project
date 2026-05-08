/*
 * transient_wall_heat_flux.c
 * Time-varying wall heat flux as a DEFINE_PROFILE on a wall boundary.
 *
 * Hook as: Boundary Conditions -> wall -> Thermal -> Heat Flux -> UDF.
 */

#include "udf.h"

#define Q0 500.0       /* W/m^2 */
#define OMEGA 6.2832   /* rad/s -> 1 Hz */

DEFINE_PROFILE(wall_heat_flux, thread, position)
{
    face_t f;
    real t = CURRENT_TIME;

    begin_f_loop(f, thread)
    {
        F_PROFILE(f, thread, position) = Q0 * (1.0 + 0.5 * sin(OMEGA * t));
    }
    end_f_loop(f, thread)
}
