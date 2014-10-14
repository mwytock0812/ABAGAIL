package opt.test;

import java.util.Arrays;

import dist.DiscreteDependencyTree;
import dist.DiscreteUniformDistribution;
import dist.Distribution;

import opt.DiscreteChangeOneNeighbor;
import opt.EvaluationFunction;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.*;
import opt.ga.CrossoverFunction;
import opt.ga.DiscreteChangeOneMutation;
import opt.ga.SingleCrossOver;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;

/**
 * Copied from ContinuousPeaksTest
 * @version 1.0
 */
public class FourPeaksTest {
    /** The n value */
    private static final int N = 200;
    /** The t value */
    private static final int T = N / 5;

    public static void main(String[] args) {
        int[] ranges = new int[N];
        Arrays.fill(ranges, 2);
        EvaluationFunction ef = new FourPeaksEvaluationFunction(T);
        Distribution odd = new DiscreteUniformDistribution(ranges);
        NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);
        MutationFunction mf = new DiscreteChangeOneMutation(ranges);
        CrossoverFunction cf = new SingleCrossOver();
        Distribution df = new DiscreteDependencyTree(.1, ranges);
        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
        ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);

        double rhcStart = System.nanoTime(), rhcEnd, rhcTime;
        RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);
        FixedIterationTrainer fit = new FixedIterationTrainer(rhc, 200000);
        fit.train();
        rhcEnd = System.nanoTime();
        rhcTime = rhcEnd - rhcStart;
        rhcTime /= Math.pow(10,9);
        System.out.println("RHC: " + ef.value(rhc.getOptimal()) + "," + rhcTime);

        double saStart = System.nanoTime(), saEnd, saTime;
        SimulatedAnnealing sa = new SimulatedAnnealing(1E11, .95, hcp);
        fit = new FixedIterationTrainer(sa, 200000);
        fit.train();
        saEnd = System.nanoTime();
        saTime = saEnd - saStart;
        saTime /= Math.pow(10,9);
        System.out.println("SA: " + ef.value(sa.getOptimal()) + "," + saTime);

        double gaStart = System.nanoTime(), gaEnd, gaTime;
        StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(200, 100, 10, gap);
        fit = new FixedIterationTrainer(ga, 1000);
        fit.train();
        gaEnd = System.nanoTime();
        gaTime = gaEnd - gaStart;
        gaTime /= Math.pow(10,9);
        System.out.println("GA: " + ef.value(ga.getOptimal()) + "," + gaTime);

        double mimicStart = System.nanoTime(), mimicEnd, mimicTime;
        MIMIC mimic = new MIMIC(200, 20, pop);
        fit = new FixedIterationTrainer(mimic, 1000);
        fit.train();
        mimicEnd = System.nanoTime();
        mimicTime = mimicEnd - mimicStart;
        mimicTime /= Math.pow(10,9);
        System.out.println("MIMIC: " + ef.value(mimic.getOptimal()) + "," + mimicTime);
    }
}
