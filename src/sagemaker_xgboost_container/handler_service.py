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

import logging
import traceback
import json
import numpy as np
from sagemaker_containers.beta.framework import encoders
from sagemaker_inference import default_inference_handler
from sagemaker_inference.default_handler_service import DefaultHandlerService
from sagemaker_inference.transformer import Transformer


from sagemaker_xgboost_container import encoder as xgb_encoders
import xgboost as xgb

from exfil_core.ml.features import calculate_features
from exfil_core.ml.prediction import decode_labels, enumerate_fields
from exfil_core.vision.objects import Phrase

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
            try:
                model_file = "xgboost-model.json"
                booster = xgb.XGBClassifier()
                booster.load_model(model_file)
                return booster
            except Exception as e:
                logging.error(traceback.format_exc())
                raise e

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
            try:
                if content_type == "application/json":
                    return json.loads(input_data)
            except Exception as e:
                logging.error(traceback.format_exc())
                raise e

            return xgb_encoders.decode(input_data, content_type)

        def default_predict_fn(self, input_data, model):
            """A default predict_fn for XGBooost Framework. Calls a model on data deserialized in input_fn.
            Args:
                input_data: input data (DMatrix) for prediction deserialized by input_fn
                model: XGBoost model loaded in memory by model_fn
            Returns: a prediction
            """
            try:

                _, _, feature_maps = calculate_features(
                    [Phrase(**phrase) for phrase in input_data['phrases']],
                    input_data['pages'],
                    input_data['model']['keywords'],
                    input_data['model']['headers'],
                    input_data['model']['global_keywords'],
                    input_data['document_type']
                )

                hashable_feature_list = tuple(
                    input_data['model']['features']
                )

                bbox_ids, X = map(
                    list,
                    zip(
                        *(
                            (f.id, f.export_features(features_to_include=hashable_feature_list))
                            for f in feature_maps
                        )
                    ),
                )

                prediction = model.predict_proba(np.array(X))
                labels = np.argmax(prediction, axis=1).tolist()
                probabilities = np.amax(prediction, axis=1).tolist()
                output = {}

                fields = enumerate_fields(input_data['model']['fields'])
                label_encoding = input_data['model']['label_encoding']
                for bbox_id, label, _ in zip(
                    bbox_ids, labels, probabilities
                ):
                    if label_encoding:
                        label = decode_labels(label, label_encoding)
                    if label > 0:
                        field_guid = [field['guid'] for field in fields if int(field['index']) == label][0]
                        output[bbox_id] = field_guid
            except Exception as e:
                logging.error(traceback.format_exc())
                raise e

            return output


        def default_output_fn(self, prediction, accept):
            """Function responsible to serialize the prediction for the response.
            Args:
                prediction (obj): prediction returned by predict_fn .
                accept (str): accept content-type expected by the client.
            Returns:
                encoded response for MMS to return to client
            """
            if accept == "application/json":
                try:
                    return json.dumps(prediction)
                except Exception as e:
                    logging.error(e)
                    raise e

            return encoders.encode(prediction, accept)

    def __init__(self):
        transformer = Transformer(default_inference_handler=self.DefaultXGBoostUserModuleInferenceHandler())
        super(HandlerService, self).__init__(transformer=transformer)
