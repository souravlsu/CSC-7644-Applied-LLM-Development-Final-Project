/*
 * temperature_dependent_viscosity.c
 * Simple Arrhenius-style temperature-dependent dynamic viscosity.
 *
 * Hook as: Materials -> <fluid> -> Viscosity -> user-defined -> mu_of_T.
 */

#include "udf.h"

#define MU_REF 1.0e-3  /* Pa*s at T_REF */
#define T_REF  300.0   /* K */
#define E_OVER_R 1500.0 /* K */

DEFINE_PROPERTY(mu_of_T, cell, thread)
{
    real T = C_T(cell, thread);
    if (T < 1.0) T = 1.0;
    return MU_REF * exp(E_OVER_R * (1.0 / T - 1.0 / T_REF));
}
