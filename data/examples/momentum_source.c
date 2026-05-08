/*
 * momentum_source.c
 * Linear momentum source term in the x-direction (drag-like resistance).
 * Returns dS/dU so Fluent can linearize implicitly.
 *
 * Hook as: Cell Zone Conditions -> fluid zone -> Source Terms -> X Momentum.
 */

#include "udf.h"

#define K_DRAG 10.0 /* 1/s */

DEFINE_SOURCE(x_momentum_source, cell, thread, dS, eqn)
{
    real u = C_U(cell, thread);
    real source = -K_DRAG * u;
    dS[eqn] = -K_DRAG;
    return source;
}
