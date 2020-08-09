from postprocessing.metrics.mase import MASE
from postprocessing.metrics.rmse import RMSE
from postprocessing.smoothing import smooth_results


def is_criteria_met(criteria, results):
    #TODO to be implemented, at the moment the algorithme only stops when MASE > 1
    return False


def update_params(params, iter):
    new_p_ab = params["p_ab_update"](iter)
    params["p_coeff"]["A"] = new_p_ab
    params["p_coeff"]["B"] = new_p_ab
    return params


def progressive_improvement_clinical_acceptability(subject, model_class, params_step1, params_step2, ph, freq, train, valid, test, scalers, file, eval_set="valid"):
    """
    For every train/valid/test split, do the PICA training algorithm :
    1. first model is trained on MSE
    2. given model is finetuned with gcMSE with progressively lower weights on A and B regions of P-EGA
    3. algorithm is stopped if MASE(results) > 1 or if criteria in params_step2 are reached (e.g, CG_EGA_AP thresholds)
    :param subject: name of subject
    :param model_class: class of model
    :param params_step1: params used during MSE fitting
    :param params_step2: params used during gcMSE finetuning
    :param ph: prediction horizon
    :param freq: prediction frequency
    :param train: train set
    :param valid: valid set
    :param test: test set
    :param scalers: scalers of train set
    :param file: basename of file to save weights
    :param eval_set: evaluation set (either "valid" or "test")
    :return: final results on eval_set, intermediate results on valid set
    """
    results_test = []
    results_valid = []

    for i, (train_i, valid_i, test_i, scaler_i) in enumerate(zip(train, valid, test, scalers)):
        mean, std = scaler_i.mean_[-2:], scaler_i.scale_[-2:]
        file_step1 = file + "_step1_" + str(i)
        file_step2 = file + "_step2_" + str(i)
        results_iter = []

        """ Step 1 - Train MSE """
        model = model_class(subject, ph, params_step1, train_i, valid_i, test_i)
        model.fit(mean, std)
        model.save(file_step1)
        results_iter.append(model.predict(dataset="valid"))

        if (MASE(results_iter[-1], ph, freq) < 1 and not is_criteria_met(params_step2["criteria"], results_iter[-1])):
            results_iter.append(smooth_results(results_iter[-1], params_step2["smoothing"]))

        """ Step 2 - Finetune gcMSE """
        iter = 1
        while(MASE(results_iter[-1], ph, freq) < 1 and not is_criteria_met(params_step2["criteria"], results_iter[-1])):
            params_iter = update_params(params_step2, iter)
            model = model_class(subject, ph, params_iter, train_i, valid_i, test_i)
            model.load(file_step1)
            model.fit(mean, std)
            results = model.predict(dataset="valid",clear=False)
            results = smooth_results(results, params_iter["smoothing"])
            results_iter.append(results)
            if MASE(results_iter[-1], ph, freq) < 1:
                model.save(file_step2) # save model only if it is acceptable (thus, last iter not saved)
                model._clear_checkpoint()
            iter += 1

        results_valid.append(results_iter)
        model.load(file_step2) if iter > 2 else model.load(file_step1)
        results = model.predict(dataset=eval_set)
        if len(results_iter) > 2:
            results = smooth_results(results, params_step2["smoothing"])
        results_test.append(results)

    return results_test, results_valid

