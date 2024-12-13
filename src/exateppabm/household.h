#pragma once
#include "flamegpu/flamegpu.h"

#include "exateppabm/input.h"

namespace exateppabm {
namespace household {

/**
 * Define flame gpu 2 model properties and functions related to households
 * 
 * @param model flamegpu2 model description object to mutate
 * @param params model parameters from parameters file
 */
void define(flamegpu::ModelDescription& model, const exateppabm::input::config& params);

/**
 * Append household related agent functions to the flame gpu control flow using layers. This is intended to be called within person::appendLayers 
 *
 * Does not use the DAG abstraction due to previously encountered bugs with split compilation units which have not yet been pinned down / resolved.
 *
 * @param model flamegpu2 model description object to mutate
 */
void appendLayers(flamegpu::ModelDescription& model);

}  // namespace household
}  // namespace exateppabm
