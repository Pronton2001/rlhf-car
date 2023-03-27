from l5kit.simulation.dataset import SimulationConfig
from l5kit.simulation.unroll import ClosedLoopSimulator
from l5kit.cle.closed_loop_evaluator import ClosedLoopEvaluator, EvaluationPlan
from l5kit.cle.metrics import (CollisionFrontMetric, CollisionRearMetric, CollisionSideMetric,
                               DisplacementErrorL2Metric, DistanceToRefTrajectoryMetric)
from l5kit.cle.validators import RangeValidator, ValidationCountingAggregator
from prettytable import PrettyTable
from l5kit.environment.gym_metric_set import L2DisplacementYawMetricSet, CLEMetricSet

def CLEValidator(sim_outs):
    metrics = [DisplacementErrorL2Metric(),
               DistanceToRefTrajectoryMetric(),
               CollisionFrontMetric(),
               CollisionRearMetric(),
               CollisionSideMetric()]

    validators = [RangeValidator("displacement_error_l2", DisplacementErrorL2Metric, max_value=30),
                  RangeValidator("distance_ref_trajectory", DistanceToRefTrajectoryMetric, max_value=4),
                  RangeValidator("collision_front", CollisionFrontMetric, max_value=0),
                  RangeValidator("collision_rear", CollisionRearMetric, max_value=0),
                  RangeValidator("collision_side", CollisionSideMetric, max_value=0)]

    intervention_validators = ["displacement_error_l2",
                               "distance_ref_trajectory",
                               "collision_front",
                               "collision_rear",
                               "collision_side"]

    cle_evaluator = ClosedLoopEvaluator(EvaluationPlan(metrics=metrics,
                                                       validators=validators,
                                                       composite_metrics=[],
                                                       intervention_validators=intervention_validators))
    
    # Calculate results
    cle_evaluator.evaluate(sim_outs)
    validation_results = cle_evaluator.validation_results()
    agg = ValidationCountingAggregator().aggregate(validation_results)
    cle_evaluator.reset()
    return agg

def quantify_outputs(sim_outs, metric_set=None):
    metric_set = metric_set if metric_set is not None else CLEMetricSet()

    metric_set.evaluate(sim_outs)
    scene_results = metric_set.evaluator.scene_metric_results
    fields = ["scene_id", "FDE", "ADE", "DRT", "CF", "CR", "CS", "PEGO"]
    table = PrettyTable(field_names=fields)
    tot_fde = 0.0
    tot_ade = 0.0
    tot_drt = 0.0
    tot_cf = 0.0
    tot_cr = 0.0
    tot_cs = 0.0
    tot_p_ego = 0.0
    tot_a_ego = 0.0
    # print(scene_results[0])
    for scene_id in scene_results:
        scene_metrics = scene_results[scene_id]
        ade_error = scene_metrics["displacement_error_l2"][1:].mean()
        fde_error = scene_metrics['displacement_error_l2'][-1]
        drt_error = scene_metrics['distance_to_reference_trajectory'][-1]
        cf_error = scene_metrics['collision_front'][-1]
        cr_error = scene_metrics['collision_rear'][-1]
        cs_error = scene_metrics['collision_side'][-1]
        p_ego = scene_metrics['simulated_minus_recorded_ego_speed'][-1]
        # a_ego = scene_metrics['aggressive_ego'][-1]
        table.add_row([scene_id, round(fde_error.item(), 4), round(ade_error.item(), 4), round(drt_error.item(), 4), round(cf_error.item(), 4), round(cr_error.item(), 4), 
        round(cs_error.item(), 4), round(p_ego.item(), 4)])
        tot_fde += fde_error.item()
        tot_ade += ade_error.item()
        tot_drt += drt_error.item()
        tot_cf += cf_error.item()
        tot_cr += cr_error.item()
        tot_cs += cs_error.item()
        tot_p_ego += p_ego.item()
        # tot_a_ego += a_ego.item()

    ave_fde = tot_fde / len(scene_results)
    ave_ade = tot_ade / len(scene_results)
    ave_drt = tot_drt / len(scene_results)
    ave_cf = tot_cf / len(scene_results)
    ave_cr = tot_cr / len(scene_results)
    ave_cs = tot_cs / len(scene_results)
    ave_p_ego = tot_p_ego / len(scene_results)
    # ave_a_ego = tot_a_ego / len(scene_results)
    table.add_row(["Overall", round(ave_fde, 4), round(ave_ade, 4), round(ave_drt, 4), round(ave_cf, 4), round(ave_cr, 4), round(ave_cs, 4), round(ave_p_ego, 4)])
    print(table)


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
