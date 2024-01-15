/*
// Copyright (C) 2021-2023 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

#include "models/model_base.h"

#include <utility>

#include <openvino/openvino.hpp>

#include <utils/common.hpp>
#include <utils/config_factory.h>
#include <utils/ocv_common.hpp>
#include <utils/slog.hpp>

std::shared_ptr<ov::Model> ModelBase::prepareModel(ov::Core& core) {
    // --------------------------- Read IR Generated by ModelOptimizer (.xml and .bin files) ------------
    /** Read model **/
    slog::info << "Reading model " << modelFileName << slog::endl;
    std::shared_ptr<ov::Model> model = core.read_model(modelFileName);
    logBasicModelInfo(model);
    // -------------------------- Reading all outputs names and customizing I/O tensors (in inherited classes)
    prepareInputsOutputs(model);
    setBatch(model);

    return model;
}

ov::CompiledModel ModelBase::compileModel(const ModelConfig& config, ov::Core& core) {
    this->config = config;
    auto model = prepareModel(core);
    compiledModel = core.compile_model(model, config.deviceName, config.compiledModelConfig);
    logCompiledModelInfo(compiledModel, modelFileName, config.deviceName);
    return compiledModel;
}

ov::Layout ModelBase::getInputLayout(const ov::Output<ov::Node>& input) {
    ov::Layout layout = ov::layout::get_layout(input);
    if (layout.empty()) {
        if (inputsLayouts.empty()) {
            layout = getLayoutFromShape(input.get_partial_shape());
            slog::warn << "Automatically detected layout '" << layout.to_string() << "' for input '"
                       << input.get_any_name() << "' will be used." << slog::endl;
        } else if (inputsLayouts.size() == 1) {
            layout = inputsLayouts.begin()->second;
        } else {
            layout = inputsLayouts[input.get_any_name()];
        }
    }

    return layout;
}

void ModelBase::setBatch(std::shared_ptr<ov::Model>& model) {
    ov::set_batch(model, 1);
}
