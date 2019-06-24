#!/usr/bin/env python

import os
import pandas
import argparse
import importlib
import pickle
import json

from sklearn.model_selection import cross_val_score


def regression(args):
    features = pandas.read_csv(args.features, sep="\t", index_col=0)
    labels = pandas.read_csv(args.labels, sep="\t", index_col=0)

    tmp = args.trainer.split(".")
    modName = ".".join(tmp[:-1])
    clsName = tmp[-1]

    mod = importlib.import_module(modName)
    cls = getattr(mod, clsName)
    f = cls()
    isect = features.index.intersection(labels.index)
    for i, c in enumerate(labels):
        X = features.loc[isect]
        y = labels[c][isect]

        scores = cross_val_score(f, X, y, cv=5, scoring="neg_mean_squared_error")

        f.fit(X, y)
        with open( os.path.join(args.outdir, "model.%d.pickle" % (i)), "w") as handle:
            pickle.dump(f, handle)
        meta = {
            "type" : "ml_model",
            "phenotype" : c,
            "training_set" : list(isect),
            "features" : list(features.columns),
            "docker" : "bmeg/vise-skrunner",
            "method" : args.trainer,
            "cross_validation" : [{
                "metric" : "neg_mean_squared_error",
                "scores" : list(scores),
            }],
        }
        with open( os.path.join(args.outdir, "model.%d.json" % (i)), "w") as handle:
            handle.write(json.dumps(meta))

def feature_select(args):
    features = pandas.read_csv(args.features, sep="\t", index_col=0)
    labels = pandas.read_csv(args.labels, sep="\t", index_col=0)

    tmp = args.trainer.split(".")
    modName = ".".join(tmp[:-1])
    clsName = tmp[-1]

    mod = importlib.import_module(modName)
    cls = getattr(mod, clsName)
    f = cls()
    isect = features.index.intersection(labels.index)
    for i, c in enumerate(labels):
        X = features.loc[isect]
        y = labels[c][isect]
        f.fit(X,y)
        s = f.get_support()

        meta = {
            "type" : "ml_featureset",
            "features" : list( X.columns[s] )
        }

        with open( os.path.join(args.outdir, "select.%d.json" % (i)), "w") as handle:
            handle.write(json.dumps(meta))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    parser_train = subparsers.add_parser('regression')
    parser_train.add_argument("features", help="Feature Matrix")
    parser_train.add_argument("labels", help="Label Matrix")
    parser_train.add_argument("--trainer", help="Training Method", default="sklearn.linear_model.Lasso")
    parser_train.add_argument("-o", "--out", dest="outdir", default="./")
    parser_train.set_defaults(func=regression)

    parser_fselect = subparsers.add_parser('feature-select')
    parser_fselect.add_argument("features", help="Feature Matrix")
    parser_fselect.add_argument("labels", help="Label Matrix")
    parser_fselect.add_argument("--trainer", help="Training Method", default="sklearn.feature_selection.SelectKBest")
    parser_fselect.add_argument("-o", "--out", dest="outdir", default="./")
    parser_fselect.set_defaults(func=feature_select)


    args = parser.parse_args()
    args.func(args)
