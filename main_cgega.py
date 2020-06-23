import sys
import argparse
from os.path import join
import misc.constants as cs
from misc.utils import printd
from misc.utils import locate_model, locate_params
from preprocessing.preprocessing import preprocessing
from processing.cross_validation import make_predictions_pclstm
from processing.cv_cgega_iterative_training import cg_ega_iterative_training
from postprocessing.postprocessing import postprocessing, postprocessing_all_iter
from postprocessing.results import ResultsSubject
from postprocessing.results_apac import ResultsSubjectAPAC
import misc

def main_cgega_iterative_training(dataset, subject, model, params1, params2, exp, eval_set, ph, save_iter=False):
    printd(dataset, subject, model, params1, params2, exp, eval_set, ph)

    # retrieve model's parameters
    params1 = locate_params(params1)
    params2 = locate_params(params2)
    model_class = locate_model(model)

    # scale variables in minutes to the benchmark sampling frequency
    ph_f = ph // cs.freq
    hist_f = params1["hist"] // cs.freq
    day_len_f = cs.day_len // cs.freq
    freq_ds = misc.datasets.datasets[dataset]["glucose_freq"]

    """ PREPROCESSING """
    train, valid, test, scalers = preprocessing(dataset, subject, ph_f, hist_f, day_len_f)

    """ MODEL TRAINING """
    dir = join(cs.path, "processing", "models", "weights", "cg_ega")
    file = join(dir, exp, model_class.__name__ + "_" + dataset + subject)

    results_test, results_valid_iter = cg_ega_iterative_training(subject, model_class, params1, params2, ph,
                                                                 freq_ds, train, valid, test, scalers, file, eval_set)

    results_test = postprocessing(results_test, scalers, dataset)
    results_valid_iter = postprocessing_all_iter(results_valid_iter, scalers, dataset)

    ResultsSubject(model, exp, ph, dataset, subject, params=[params1, params2], results=results_test).save_raw_results()
    if save_iter:
        ResultsSubjectAPAC(model, exp, ph, dataset, subject, params=[params1, params2], results=results_valid_iter).save_raw_results()

def main_standard(dataset, subject, model, params, exp, eval_set, ph):
    printd(dataset, subject, model, params, exp, eval_set, ph)

    # retrieve model's parameters
    params = locate_params(params)
    model_class = locate_model(model)

    # scale variables in minutes to the benchmark sampling frequency
    ph_f = ph // cs.freq
    hist_f = params["hist"] // cs.freq
    day_len_f = cs.day_len // cs.freq

    """ PREPROCESSING """
    train, valid, test, scalers = preprocessing(dataset, subject, ph_f, hist_f, day_len_f)

    """ MODEL TRAINING """
    raw_results = make_predictions_pclstm(subject, model_class, params, ph_f, train, valid, test, scalers, mode=eval_set)
    """ POST-PROCESSING """
    raw_results = postprocessing(raw_results, scalers, dataset)

    """ EVALUATION """
    ResultsSubject(model, exp, ph, dataset, subject, params=params, results=raw_results).save_raw_results()

if __name__ == "__main__":
    """ The main function contains the following optional parameters:
            --mode: either "standard" or "iterative", if "iterative" it uses the APAC training algorithm;
            --dataset: which dataset to use, should be referenced in misc/datasets.py;
            --subject: which subject to use, part of the dataset, use the spelling in misc/datasets.py;
            --model: model on which the benchmark will be run (e.g., "svr"); need to be lowercase; 
            --params1: parameters of the model, usually has the same name as the model (e.g., "svr"); need to be lowercase; 
            --params2: same as params1 but is used only in "iterative" mode, during finetuning at stage 2; 
            --ph: the prediction horizon of the models; default 30 minutes;
            --exp: experimental folder in which the data will be stored, inside the results directory;
            --eval_set: specify is the model is tested on the validation "valid" set or testing "test" set ;
            --save_iter: either 0 or 1, says if the results of all iteration of APÄC must be saved

        Example:
            python main.py --dataset=ohio --subject=559 --model=base --params=base --ph=30 
                        --exp=myexp --mode=valid --plot=1 --log=mylog
    """

    """ ARGUMENTS HANDLER """
    # retrieve and process arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--subject", type=str)
    parser.add_argument("--model", type=str)
    parser.add_argument("--params1", type=str)
    parser.add_argument("--params2", type=str)
    parser.add_argument("--ph", type=int)
    parser.add_argument("--exp", type=str)
    parser.add_argument("--eval_set", type=str)
    parser.add_argument("--log", type=str)
    parser.add_argument("--save_iter", type=int)
    args = parser.parse_args()

    # compute stdout redirection to log file
    if args.log:
        sys.stdout = open(join(cs.path, "logs", args.log + ".log"), "w")

    if args.mode == "iterative":
        main_cgega_iterative_training(
            dataset=args.dataset, subject=args.subject, model=args.model, params1=args.params1,
            params2=args.params2, exp=args.exp, eval_set=args.eval_set, ph=args.ph, save_iter=args.save_iter)
    else :
        main_standard(dataset=args.dataset, subject=args.subject, model=args.model, params=args.params1,
            exp=args.exp, eval_set=args.eval_set, ph=args.ph)
