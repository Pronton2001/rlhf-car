from l5kit.simulation.dataset import SimulationConfig
from l5kit.simulation.unroll import ClosedLoopSimulator
from l5kit.cle.closed_loop_evaluator import ClosedLoopEvaluator, EvaluationPlan
from l5kit.cle.metrics import (CollisionFrontMetric, CollisionRearMetric, CollisionSideMetric,
                               DisplacementErrorL2Metric, DistanceToRefTrajectoryMetric, SimulatedDrivenMilesMetric, ReplayDrivenMilesMetric, SimulatedVsRecordedEgoSpeedMetric, SimulatedEgoAccMetric)
from l5kit.cle.validators import RangeValidator, ValidationCountingAggregator
from prettytable import PrettyTable
from l5kit.environment.gym_metric_set import L2DisplacementYawMetricSet, CLEMetricSet
import numpy as np
from l5kit.cle import metric_set, metrics, validators

def CLEValidator(sim_outs):
    metrics = [DisplacementErrorL2Metric(),
               DistanceToRefTrajectoryMetric(),
               CollisionFrontMetric(),
               CollisionRearMetric(),
               CollisionSideMetric(),
                SimulatedVsRecordedEgoSpeedMetric(),
                SimulatedDrivenMilesMetric(),
                ReplayDrivenMilesMetric(),
                # SimulatedEgoAccMetric(),
               ]

    validator_list = [RangeValidator("displacement_error_l2", DisplacementErrorL2Metric, max_value=30),
                  RangeValidator("distance_ref_trajectory", DistanceToRefTrajectoryMetric, max_value=4),
                  RangeValidator("collision_front", CollisionFrontMetric, max_value=0),
                  RangeValidator("collision_rear", CollisionRearMetric, max_value=0),
                  RangeValidator("collision_side", CollisionSideMetric, max_value=0),
                # Passiveness indicator - slow_driving metric - Failure if simulated ego is slower than recording by more
                # than 5 m/s (~11 MPH) for 2.3 seconds
                    RangeValidator("passive_ego", SimulatedVsRecordedEgoSpeedMetric,
                                      min_value=-5.0, violation_duration_s=2.3,
                                      duration_mode=validators.DurationMode.CONTINUOUS),
                    # Aggressiveness metrics - Failure if simulated ego is faster than recording by more
                    # than 5 m/s (~11 MPH) for 2.3 seconds
                    RangeValidator("aggressive_ego", SimulatedVsRecordedEgoSpeedMetric,
                                      max_value=5.0, violation_duration_s=2.3,
                                      duration_mode=validators.DurationMode.CONTINUOUS),
                    # RangeValidator("acc", SimulatedEgoAccMetric,
                    #         max_value=2.0, violation_duration_s=2.3,
                    #         duration_mode=validators.DurationMode.CONTINUOUS),
        ] 

    intervention_validators = [
                            #     'displacement_error_l2' ,
                            #     'distance_ref_trajectory',
                            #    "collision_front",
                            #    "collision_rear",
                            #    "collision_side",
    ]
                               
                               

    cle_evaluator = ClosedLoopEvaluator(EvaluationPlan(metrics=metrics,
                                                       validators=validator_list,
                                                       composite_metrics=[],
                                                       intervention_validators=intervention_validators))
    
    # Calculate results
    cle_evaluator.evaluate(sim_outs)
    validation_results = cle_evaluator.validation_results()
    agg = ValidationCountingAggregator().aggregate(validation_results)
    cle_evaluator.reset()
    fields = ["metric", "value"]
    table = PrettyTable(field_names=fields)

    values = []
    names = []

    for metric_name in agg:
        table.add_row([metric_name, agg[metric_name].item()])
        values.append(agg[metric_name].item())
        names.append(metric_name)

    print(table)
    return values

def quantify_outputs(sim_outs, metric_set=None):
    metric_set = metric_set if metric_set is not None else CLEMetricSet()

    metric_set.evaluate(sim_outs)
    scene_results = metric_set.evaluator.scene_metric_results
    fields = ['id', "scene_id", "ADE", "FDE", "D2R", "CF", "CR", "CS", "PEGO"]
    table = PrettyTable(field_names=fields)
    tot_fde = []
    tot_ade = []
    tot_d2r = []
    tot_cf = []
    tot_cr = []
    tot_cs = []
    tot_p_ego = []
    # tot_acc_ego = []
    # print(scene_results[0])
    for id, scene_id in enumerate(scene_results):
        scene_metrics = scene_results[scene_id]
        ade_error = scene_metrics["displacement_error_l2"][1:].mean()
        fde_error = scene_metrics['displacement_error_l2'][-1]
        d2r_error = scene_metrics['distance_to_reference_trajectory'][1:].mean()
        cf_error = scene_metrics['collision_front'][1:].sum()
        cr_error = scene_metrics['collision_rear'][1:].sum() 
        cs_error = scene_metrics['collision_side'][1:].sum() 
        p_ego = scene_metrics['simulated_minus_recorded_ego_speed'][1:].mean()
        # acc_ego = scene_metrics['simulated_ego_acceleration'][1:].mean()
        table.add_row([id, scene_id, round(fde_error.item(), 4), round(ade_error.item(), 4), round(d2r_error.item(), 4), round(cf_error.item(), 4), round(cr_error.item(), 4), 
        round(cs_error.item(), 4), round(p_ego.item(), 4)])
        
        tot_fde.append(fde_error.item())
        tot_ade.append(ade_error.item())
        tot_d2r.append(d2r_error.item())
        tot_cf.append(cf_error.item())
        tot_cr.append(cr_error.item())
        tot_cs.append(cs_error.item())
        tot_p_ego.append(p_ego.item())

    tot_fde = np.asarray(tot_fde)
    tot_ade = np.asarray(tot_ade)
    tot_d2r = np.asarray(tot_d2r)
    tot_cf = np.asarray(tot_cf)
    tot_cr = np.asarray(tot_cr)
    tot_cs = np.asarray(tot_cs)
    tot_p_ego = np.asarray(tot_p_ego)
    # tot_acc_ego = np.asarray(tot_acc_ego)
    
    tot = [tot_fde, tot_ade, tot_d2r, tot_cf, tot_cr, tot_cs, tot_p_ego]
    table.add_row(["Mean", '',round(tot_fde.mean(), 1), round(tot_ade.mean(), 1), round(tot_d2r.mean(), 1), round(tot_cf.mean(), 1), round(tot_cr.mean(), 1), round(tot_cs.mean(), 1), round(tot_p_ego.mean(), 1)])
    table.add_row(["Std", '', round(tot_fde.std(), 1), round(tot_ade.std(), 1), round(tot_d2r.std(), 1), round(tot_cf.std(), 1), round(tot_cr.std(), 1), round(tot_cs.std(), 1), round(tot_p_ego.std(), 1)])
    print(table)
    # print(f'{round(tot_fde.mean(), 1)} $\pm$ {round(tot_fde.std(), 1)}, \
    #         {round(tot_ade.mean(), 1)} $\pm$ {round(tot_ade.std(), 1)}, \
    #         {round(tot_d2r.mean(), 1)} $\pm$ {round(tot_d2r.std(), 1)} \
    #         {round(tot_cf.mean(), 1)} $\pm$ {round(tot_cf.std(), 1)}, \
    #         {round(tot_cr.mean(), 1)} $\pm$ {round(tot_cr.std(), 1)}, \
    #         {round(tot_cs.mean(), 1)} $\pm$ {round(tot_cs.std(), 1)}, \
    #         {round(tot_p_ego.mean(), 1)} $\pm$ {round(tot_p_ego.std(), 1)} \
    #         {round(tot_acc_ego.mean(), 1)} $\pm$ {round(tot_acc_ego.std(), 1)}' )
    # ret = [str(round(i.mean(), 1)) + ' $\pm$ '+ str(round(i.std(), 1)) + ', ' for i in tot]
    # print(*ret)
    ret = [round(i.mean(), 1) for i in tot]
    print(ret)
    return ret
# def acc(sim_outs):
#     speeds = []
#     for t in in_outs:
#         speeds.append(t.inputs['speed'])
#     # output_logits = inverseUnicycle(pred_x, pred_y, pred_yaw,
#     speeds = np.asarray(speeds)
#     acc = speeds[1:] - speeds[:-1]
#     acc = abs(acc)
STEP_TIME = 0.1
def quantitative(sim_outs, actions):
    metrics = quantify_outputs(sim_outs) # ade, fde, d2r, cf, cr, cs, pego
    val = CLEValidator(sim_outs)
    I1K = sum(val[1:5]) # d2r, cf, cr, cs (1,2,3,4)
    # steers = np.asarray(actions)[:,0]
    accs = np.asarray(actions)[:,1]
    abs_accs = abs(accs)
    ret = metrics[:3] + val[:2] + metrics[3:6] + val[2:5] + metrics[6:] + [round(abs_accs.mean(),4)] + val[5:] + [len(abs_accs[abs_accs > 2 * STEP_TIME ])] + [I1K]
    print(ret)
    return ret


def compute_ade_fde(sim_outs, metric_set=None):
    ades = []
    fdes = []
    metric_set = metric_set if metric_set is not None else L2DisplacementYawMetricSet()
    metric_set.evaluate(sim_outs)
    scene_results = metric_set.evaluator.scene_metric_results
    for scene_id in scene_results:
        scene_metrics = scene_results[scene_id]
        ade_error = scene_metrics["displacement_error_l2"][1:].mean()
        fde_error = scene_metrics['displacement_error_l2'][-1]
        ades.append(ade_error.item())
        fdes.append(fde_error.item())
    return ades, fdes   

import pickle
def save_data(data, file):
    with open(file, "wb") as f:
        pickle.dump(data, f)
# save_data(sim_outs, f'{SRC_PATH}/src/validate/sac_vector_RLfinetune_freeze_checkpoint570(-47).obj')