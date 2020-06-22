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


def cg_ega_iterative_training(subject, model_class, params_step1, params_step2, ph, freq, train, valid, test, scalers, file, eval_set="valid"):
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

        """ Step 2 - Finetune gcMSE """
        iter = 1
        while(MASE(results_iter[-1], ph, freq) < 1 and not is_criteria_met(params_step2["criteria"], results_iter[-1])):
            params_iter = update_params(params_step2, iter)
            model = model_class(subject, ph, params_iter, train_i, valid_i, test_i)
            model.load(file_step1)
            model.fit(mean, std)
            model.save(file_step2)
            results = model.predict(dataset="valid")
            results = smooth_results(results, params_iter["smoothing"])
            results_iter.append(results)
            iter += 1

        results_valid.append(results_iter)
        model.load(file_step2)
        results = model.predict(dataset=eval_set)
        results = smooth_results(results, params_step2["smoothing"])
        results_test.append(results)

    return results_test, results_valid



# def two_step_training(subject, model_class, params_step1, params_step2, ph, freq, train, valid, test, scalers, file, mode="valid"):
#     results = []
#     for i, (train_i, valid_i, test_i, scaler_i) in enumerate(zip(train, valid, test, scalers)):
#         mean, std = scaler_i.mean_[-2:], scaler_i.scale_[-2:]
#         results_cv = []
#
#         printd(" --- STEP 1 ---")
#         model = model_class(subject, ph, params_step1, train_i, valid_i, test_i)
#         model.fit(mean, std)
#         weights_file_step1 = file + "_step1_" + str(i)
#         model.save(weights_file_step1)
#         results_cv.append(model.predict(dataset=mode))
#         printd("MASE =",MASE(results_cv[-1], ph, freq))
#
#         printd("--- STEP 2 ---")
#         iter = 1
#
#         # while (MASE(results_cv[-1], ph, freq) < 1) or iter < 15:
#         while iter < 30:
#             params_step2_iter = copy.deepcopy(params_step2)
#             params_step2_iter["p_coeff"]["A"], params_step2_iter["p_coeff"]["uB"], params_step2_iter["p_coeff"]["lB"] = 1/iter, 1/iter, 1/iter
#
#             model = model_class(subject, ph, params_step2_iter, train_i, valid_i, test_i)
#             model.load(weights_file_step1)
#             model.fit(mean, std)
#             weights_file_step2 = file + "_step2-" + str(iter) + "_" + str(i)
#             model.save(weights_file_step2)
#             results_cv.append(model.predict(dataset=mode))
#             printd("iter =",iter, "MASE =",MASE(results_cv[-1], ph, freq))
#
#             iter += 1
#             # break
#
#         results.append(results_cv)
#
#
#     return results
