(ns ^{:doc "Some functions to help test the algorithms using a monte carlo simulation."
      :author "Paul Ingles"}
    bandit.simulate
  (:use [clojure.data.csv :only (write-csv)]
        [clojure.java.io :only (writer)]
        [clojure.string :only (join)]
        [clojure.java.io :only (writer)]
        [bandit.arms :only (update pulled reward)]
        [clojure.tools.cli :only (cli)]
        [incanter.stats :only (mean)])
  (:require [bandit.algo.exp3 :as exp3]
            [bandit.algo.bayes :as bayes]
            [bandit.algo.ucb :as ucb]
            [bandit.algo.softmax :as softmax]
            [bandit.test-cases :as test-cases]
            [quil.core :as q])
  (:gen-class))

(defn bernoulli-arm
  "creates a function representing the bandit arm. uses a fixed
   probability p of reward. p of 0.1 would reward ~10% of the time."
  [p]
  (fn []
    (if (> (rand) p)
      0
      1)))

(defn draw-arm [f] (f))

(defn mk-bernoulli-bandit
  "creates the simulation bandit: a map of labels to their arms (functions)"
  [& p]
  (->> p
       (partition 2)
       (map (fn [[label pct]] {label (bernoulli-arm pct)}))
       (reduce merge)))





(defn simulate
  "runs a simulation. bandit is a sequence of arms (functions) that
   return their numerical reward value.
   bandit: the multi-armed machine we're optimising against
   selectfn: the algorithm function to select the arm. (f arms)
   arms: current algorithm state"
  [bandit selectfn rewardfn {:keys [arms result]}]
  (let [gamma 0.1
        pull (selectfn (vals arms))
        selected-label (:name pull)
        arm (get bandit selected-label)
        rwd (draw-arm arm)
        {:keys [cumulative-reward t]} result]
    {:arms (update (-> pull (rewardfn rwd) (pulled)) arms)
     :result {:pulled selected-label
              :reward rwd
              :t (inc t)
              :cumulative-reward (+ cumulative-reward rwd)}}))

(defn simulation-seq
  "returns an unbounded sequence with results and arms for a test run.
   the number of items taken represents the horizon (or t) value.

   example: to run the algorithm against the bandit to horizon 20:

   (take 20 (simulation-seq bandit bayes/select-arm bayes/reward arms))"
  [bandit selectfn rewardfn arms]
  (rest (iterate (partial simulate bandit selectfn rewardfn)
                 {:arms arms
                  :result {:t 0
                           :cumulative-reward 0}})))

(defn- summary-row
  "Results is the set of results at time t across all simulations."
  [results]
  (let [t (:t (first results))]
    ["bayes" "bayes" "0.3" 1 t (:pulled (first results)) (:reward (first results))
     (float (mean (map :cumulative-reward results)))]))

(defn transpose
  [coll]
  (partition (count coll)
             (apply interleave coll)))

(defn -main
  [& args]
  (let [[options args banner] (cli args
                                   ["-a" "--algorithm" "Algorithm to use." :default "bayes"]
                                   ["-e" "--epsilon" "Value to control algorithm's tendency to explore." :default 1.0]
                                   ["-o" "--output" "File path to write results to" :default "results.csv"]
                                   ["-n" "--num-simulations" "Number of simulations to execute" :default 1]
                                   ["-t" "--time" "Time: number of iterations within simulation" :default 27332]
                                   ["-h" "--help" "Display this"])]
    (when (:help options)
      (println banner)
      (System/exit 0))
    (time
     (let [{:keys [output num-simulations time algorithm epsilon]} options
           algo        {:select bayes/select-arm
                        :reward bayes/reward}]
       (println "Starting simulations ...")
       (with-open [out-csv (writer "output.csv")]
         (write-csv out-csv
                    (doall
                     (for [[control-samples variant-samples control-conversions variant-conversions]
                           test-cases/realistic-case]
                       (let [control-conversion (/ (double control-conversions) (double control-samples))
                             variant-conversion (/ (double variant-conversions) (double variant-samples))
                             number-of-samples  (+ control-samples variant-samples)
                             bandit      (mk-bernoulli-bandit :control control-conversion :variant variant-conversion)
                             arms        (exp3/bandit :control :variant)]
                         (println "Running test with " control-samples " control samples and " variant-samples " variant samples")
                         (into [control-samples variant-samples control-conversions variant-conversions (+ control-conversions variant-conversions) control-conversion variant-conversion]
                               (doall (for [i (range 10)]
                                        (->> (simulation-seq bandit
                                                             bayes/select-arm
                                                             bayes/reward
                                                             arms)
                                             (take number-of-samples)
                                             (map :result)
                                             (sort-by :t)
                                             last
                                             :cumulative-reward)))))))))
       (println "Completed simulations. Results in" output)))))



(defn simulate-chained-bandit
  "runs a simulation. bandit is a sequence of arms (functions) that
   return their numerical reward value.
   bandit: the multi-armed machine we're optimising against
   selectfn: the algorithm function to select the arm. (f arms)
   arms: current algorithm state"
  [{:keys [arms result bandit]}]
  (let [[home-page-arms results-page-arms] arms
        [home-page-bandit results-page-bandit] bandit
        first-pull   (bayes/select-arm (vals home-page-arms))
        second-pull  (bayes/select-arm (vals results-page-arms))
        first-label  (:name first-pull)
        second-label (:name second-pull)
        rwd (* (draw-arm (get home-page-bandit first-label))
               (draw-arm (get results-page-bandit second-label)))
        {:keys [cumulative-reward t]} result]
    {:arms [(update (-> first-pull (bayes/reward rwd) (pulled)) home-page-arms)
            (update (-> second-pull (bayes/reward rwd) (pulled)) results-page-arms)]
     :bandit bandit
     :result {:pulled [first-pull second-pull]
              :reward rwd
              :t (inc t)
              :cumulative-reward (+ cumulative-reward rwd)}}))

;; (defn simulate
;;   "runs a simulation. bandit is a sequence of arms (functions) that
;;    return their numerical reward value.
;;    bandit: the multi-armed machine we're optimising against
;;    selectfn: the algorithm function to select the arm. (f arms)
;;    arms: current algorithm state"
;;   [bandit selectfn rewardfn {:keys [arms result]}]
;;   (let [gamma 0.1
;;         pull (selectfn (vals arms))
;;         selected-label (:name pull)
;;         arm (get bandit selected-label)
;;         rwd (draw-arm arm)
;;         {:keys [cumulative-reward t]} result]
;;     {:arms (update (-> pull (rewardfn rwd) (pulled)) arms)
;;      :result {:pulled selected-label
;;               :reward rwd
;;               :t (inc t)
;;               :cumulative-reward (+ cumulative-reward rwd)}}))


(def initial-state
  {:bandit [{:home-page-variant-a (bernoulli-arm 0.1)
             :home-page-variant-b (bernoulli-arm 0.3)}
            {:results-page-variant-a (bernoulli-arm 0.9)
             :results-page-variant-b (bernoulli-arm 0.2)}]
   :arms   [(exp3/bandit :home-page-variant-a
                         :home-page-variant-b)
            (exp3/bandit :results-page-variant-a
                         :results-page-variant-b)]
   :result {:pulled nil
            :reward 0
            :t 0
            :cumulative-reward 0}})

(defn run-chained-simulation
  []
  (iterate simulate-chained-bandit initial-state))

(def arm-paths
  [[0 :home-page-variant-a]
   [0 :home-page-variant-b]
   [1 :results-page-variant-a]
   [1 :results-page-variant-b]])

(defn prepare-data
  []
  (->> (run-chained-simulation)
       (map (fn [{:keys [arms result]}]
              (reduce concat
                      (into [((juxt :t (comp :name :pulled) :reward :cumulative-reward) result)]
                            (for [path arm-paths]
                              (let [arm (get-in arms path)]
                                [(:name arm) (:pulls arm) (:value arm) (:weight arm)]))))))
       (take 100000)))

(defn calculate-conversions-under-chained-methodology
  [number-of-samples]
  (doall (for [i (range 100)]
           (->> (run-chained-simulation)
                (take number-of-samples)
                (map :result)
                (sort-by :t)
                last
                :cumulative-reward))))

(defn write-to-csv
  [data]
  (with-open [out-csv (writer "/tmp/output.csv")]
    (write-csv out-csv data)))

(def current-state
  (atom initial-state))

(defn update-state!
  []
  (println "-------------------------------------------")
  (println @current-state)
  (swap! current-state simulate-chained-bandit))

(def initial-position
  {:home-page-variant-a [20 20]
   :home-page-variant-b [20 100]
   :results-page-variant-a [100 20]
   :results-page-variant-b [100 100]})

(defn draw
  []
  (update-state!)
  (doseq [arm-path arm-paths]
    (let [arm (get-in (:arms @current-state) arm-path)
          [x y] (initial-position (:name arm))]
      (println (:pulls arm))
      (q/ellipse x y (+ 5 (:pulls arm)) (+ 5 (:pulls arm))))))

(defn setup
  []
  (q/frame-rate 5))

(defn draw-it
  []
  (q/defsketch bandit-sketch                  
    :title    "Chained Bandit"    
    :settings #(q/smooth 2)             
    :draw     draw
    :setup    setup
    :size     [323 200]))
