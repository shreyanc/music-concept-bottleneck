import torch
import torch.nn as nn
import torch.nn.functional as F
import models.model_utils as model_utils
import logging

logger = logging.getLogger()

from models.cpresnet import CPResnet, ReverseLayerF
from models.model_configs import config_cp_field_shallow_m2


class CPResnetMultihead(CPResnet):
    def __init__(self, tasks_outputs=None, **kwargs):
        super(CPResnetMultihead, self).__init__(config=config_cp_field_shallow_m2, num_targets=2, ff=None, **kwargs)
        if tasks_outputs is None:
            tasks_outputs = {'av': 2}

        self.heads = nn.ModuleDict({})
        for task, num_outputs in tasks_outputs.items():
            self.heads[task] = self._make_head(self.n_channels[2], num_outputs)

        self.apply(self._initialize_weights)

    def forward(self, x, **kwargs):
        features = self.forward_conv(x)

        outs = {}
        for task, head in self.heads.items():
            outs[task] = head(features)

        return {"output": outs, "embedding": features}

    def _make_head(self, num_in, num_out):
        head = nn.Sequential(nn.Conv2d(num_in,
                                       num_out,
                                       kernel_size=1,
                                       stride=1,
                                       padding=0,
                                       bias=False),
                             nn.BatchNorm2d(num_out),
                             self.pool,
                             nn.Flatten()
                             )
        return head


class CPResnetExplainable(CPResnet):
    def __init__(self, config=None, num_interpretable_concepts=None, num_targets=None, ff_hidden_size=None, **kwargs):
        if config is None:
            config = config_cp_field_shallow_m2

        # if (concept_input_bias := kwargs.pop("concept_input_bias")) is None:
        #     concept_input_bias = False

        if (concept_input_bn := kwargs.pop("concept_input_bn", None)) is None:
            concept_input_bn = True

        if concept_input_bn:
            concept_input_bias = False
        else:
            concept_input_bias = True

        if (concept_layer_type := kwargs.pop("concept_layer_type", None)) is None:
            concept_layer_type = 'conv'

        if (ff_bias := kwargs.pop("ff_bias", None)) is None:
            ff_bias = True

        super(CPResnetExplainable, self).__init__(config=config, num_targets=num_targets, ff=None, **kwargs)

        if concept_layer_type == 'linear':
            if concept_input_bn:
                self.concepts_layer = nn.Sequential(
                    self.pool,
                    nn.Flatten(),
                    nn.Linear(self.n_channels[2], num_interpretable_concepts, bias=concept_input_bias),
                    nn.BatchNorm2d(num_interpretable_concepts)
                )
            else:
                self.concepts_layer = nn.Sequential(
                    self.pool,
                    nn.Flatten(),
                    nn.Linear(self.n_channels[2], num_interpretable_concepts, bias=concept_input_bias),
                )
        else:
            if concept_input_bn:
                self.concepts_layer = nn.Sequential(
                    nn.Conv2d(self.n_channels[2],
                              num_interpretable_concepts,
                              kernel_size=1,
                              stride=1,
                              padding=0,
                              bias=concept_input_bias),
                    nn.BatchNorm2d(num_interpretable_concepts),
                    self.pool,
                    nn.Flatten()
                )
            else:
                self.concepts_layer = nn.Sequential(
                    nn.Conv2d(self.n_channels[2],
                              num_interpretable_concepts,
                              kernel_size=1,
                              stride=1,
                              padding=0,
                              bias=concept_input_bias),
                    self.pool,
                    nn.Flatten()
                )

        if ff_hidden_size:
            self.feed_forward = nn.Sequential(
                nn.Linear(num_interpretable_concepts, ff_hidden_size, bias=True),
                nn.ReLU(),
                nn.Linear(ff_hidden_size, num_targets, bias=True),
            )
        else:
            self.feed_forward = nn.Linear(num_interpretable_concepts, num_targets, bias=ff_bias)

    def forward(self, x, **kwargs):
        features = self.forward_conv(x)
        concepts = self.concepts_layer(features)
        outs = self.feed_forward(concepts)
        return {"output": outs, "embedding": features, "concepts": concepts}

    @property
    def name(self):
        return "CPResnet_Explainable"


class CPResnetExplainableDA(CPResnet):
    def __init__(self, num_interpretable_concepts=None, num_targets=None, concept_layer_type='conv', **kwargs):
        super(CPResnetExplainableDA, self).__init__(config=config_cp_field_shallow_m2, num_targets=num_targets, ff=None, **kwargs)

        if concept_layer_type == 'linear':
            self.concepts_layer = nn.Sequential(
                self.pool,
                nn.Flatten(),
                nn.Linear(self.n_channels[2], num_interpretable_concepts, bias=False)
            )
        else:
            self.concepts_layer = nn.Sequential(
                nn.Conv2d(self.n_channels[2],
                          num_interpretable_concepts,
                          kernel_size=1,
                          stride=1,
                          padding=0,
                          bias=False),
                nn.BatchNorm2d(num_interpretable_concepts),
                self.pool,
                nn.Flatten()
            )

        self.discriminator = nn.Sequential(
            nn.Linear(512, 50),
            nn.ReLU(),
            nn.Linear(50, 20),
            nn.ReLU(),
            nn.Linear(20, 1)
        )

        self.feed_forward = nn.Linear(num_interpretable_concepts, num_targets, bias=False)

    def forward(self, x, **kwargs):
        num_labeled = kwargs.get('num_labeled')
        if num_labeled is None:
            num_labeled = x.shape[0]

        features = self.forward_conv(x)
        features_pooled = self.pool(features)

        domain = self.discriminator(ReverseLayerF.apply(features_pooled.view(features_pooled.size(0), -1), kwargs.get('lambda_', 1.)))
        concepts = self.concepts_layer(features)
        outs = self.feed_forward(concepts)

        return {"output": outs, "embedding": features, "concepts": concepts, "domain": domain}

    @property
    def name(self):
        return "CPResnet_ExplainableDA"


class CPResnetExplainableMLInsSigmoid(CPResnet):
    def __init__(self, config=None, num_interpretable_concepts=None, num_targets=None, **kwargs):
        if config is None:
            config = config_cp_field_shallow_m2

        super().__init__(config=config, num_targets=num_targets, ff=None, **kwargs)

        self.to_ml = nn.Sequential(
            nn.Conv2d(self.n_channels[2],
                      7,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=True),
            nn.BatchNorm2d(7),
            self.pool,
            nn.Flatten()
        )

        self.to_ins = nn.Sequential(
            nn.Conv2d(self.n_channels[2],
                      41,
                      kernel_size=1,
                      stride=1,
                      padding=0,
                      bias=True),
            nn.BatchNorm2d(41),
            self.pool,
            nn.Flatten(),
        )

        num_extranodes = num_interpretable_concepts - 48
        self.has_extranodes = False
        if num_extranodes > 0:
            self.has_extranodes = True
            self.to_extra = nn.Sequential(
                nn.Conv2d(self.n_channels[2],
                          num_extranodes,
                          kernel_size=1,
                          stride=1,
                          padding=0,
                          bias=True),
                nn.BatchNorm2d(num_extranodes),
                self.pool,
                nn.Flatten()
            )

        self.feed_forward = nn.Linear(num_interpretable_concepts, num_targets, bias=True)

    def forward(self, x, **kwargs):
        features = self.forward_conv(x)
        mls = self.to_ml(features)
        ins = self.to_ins(features)
        if self.has_extranodes:
            extra = self.to_extra(features)
            concepts = torch.hstack([mls, torch.sigmoid(ins), extra])
            concepts_no_sig = torch.hstack([mls, ins, extra])
            outs = self.feed_forward(concepts)
            return {"output": outs, "embedding": features, "concepts": concepts_no_sig, "mls": mls, "ins": torch.sigmoid(ins), "extra": extra}
        else:
            concepts = torch.hstack([mls, torch.sigmoid(ins)])
            concepts_no_sig = torch.hstack([mls, ins])
            outs = self.feed_forward(concepts)
            return {"output": outs, "embedding": features, "concepts": concepts_no_sig, "mls": mls, "ins": torch.sigmoid(ins)}

    @property
    def name(self):
        return "CPResnetExplainableMLInsSigmoid"
