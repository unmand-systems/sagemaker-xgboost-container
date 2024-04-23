# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
from __future__ import absolute_import

import json
import numpy as np
from sagemaker_containers.beta.framework import encoders
from sagemaker_inference import content_types, default_inference_handler
from sagemaker_inference.default_handler_service import DefaultHandlerService
from sagemaker_inference.transformer import Transformer


from sagemaker_xgboost_container import encoder as xgb_encoders
import xgboost as xgb

class HandlerService(DefaultHandlerService):
    """Handler service that is executed by the model server.
    Determines specific default inference handlers to use based on the type MXNet model being used.
    This class extends ``DefaultHandlerService``, which define the following:
        - The ``handle`` method is invoked for all incoming inference requests to the model server.
        - The ``initialize`` method is invoked at model server start up.
    Based on: https://github.com/awslabs/mxnet-model-server/blob/master/docs/custom_service.md
    """

    class DefaultXGBoostUserModuleInferenceHandler(default_inference_handler.DefaultInferenceHandler):
        def default_model_fn(self, model_dir):
            """Load a model. For XGBoost Framework, a default function to load a model is not provided.
            Users should provide customized model_fn() in script.
            Args:
                model_dir: a directory where model is saved.
            Returns: A XGBoost model.
            """
            model_file = "xgboost-model.json"
            booster = xgb.XGBClassifier()
            booster.load_model(model_file)
            return booster

        def default_input_fn(self, input_data, content_type):
            """Take request data and de-serializes the data into an object for prediction.
                When an InvokeEndpoint operation is made against an Endpoint running SageMaker model server,
                the model server receives two pieces of information:
                    - The request Content-Type, for example "application/json"
                    - The request data, which is at most 5 MB (5 * 1024 * 1024 bytes) in size.
                The input_fn is responsible to take the request data and pre-process it before prediction.
            Args:
                input_data (obj): the request data.
                content_type (str): the request Content-Type.
            Returns:
                (obj): data ready for prediction. For XGBoost, this defaults to DMatrix.
            """
            if content_type == "application/json":
                return np.array(json.loads(input_data))
            return xgb_encoders.decode(input_data, content_type)

        def default_predict_fn(self, input_data, model):
            """A default predict_fn for XGBooost Framework. Calls a model on data deserialized in input_fn.
            Args:
                input_data: input data (DMatrix) for prediction deserialized by input_fn
                model: XGBoost model loaded in memory by model_fn
            Returns: a prediction
            """
            return model.predict_proba(input_data)


        def default_output_fn(self, prediction, accept):
            """Function responsible to serialize the prediction for the response.
            Args:
                prediction (obj): prediction returned by predict_fn .
                accept (str): accept content-type expected by the client.
            Returns:
                encoded response for MMS to return to client
            """
            if accept == "application/json":
                labels = np.argmax(prediction, axis=1)
                probabilities = np.amax(prediction, axis=1)
                return json.dumps({
                        'labels': labels.tolist(),
                        'probabilities': probabilities.tolist(),
                    })

            encoded_prediction = encoders.encode(prediction, accept)
            if accept == content_types.CSV:
                encoded_prediction = encoded_prediction.encode("utf-8")

            return encoded_prediction

    def __init__(self):
        transformer = Transformer(default_inference_handler=self.DefaultXGBoostUserModuleInferenceHandler())
        super(HandlerService, self).__init__(transformer=transformer)
