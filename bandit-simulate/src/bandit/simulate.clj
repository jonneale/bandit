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
            [quil.core :as q]
            [quil.middleware :as m])
  (:gen-class))

(defn bernoulli-arm
  "creates a function representing the bandit arm. uses a fixed
   probability p of reward. p of 0.1 would reward ~10% of the time."
  [p]
  (fn []
    (if (> (rand) p)
      0
      1)))

(defn draw-arm [f & args] (apply f args))

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
  "retur3ns an unbounded sequence with results and arms for a test run.
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

(defn -main2
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
        rwd          (draw-arm (get home-page-bandit first-label) second-label)
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


(defn simulation-seq-simple
  "returns an unbounded sequence with results and arms for a test run.
   the number of items taken represents the horizon (or t) value.

   example: to run the algorithm against the bandit to horizon 20:

   (take 20 (simulation-seq bandit bayes/select-arm bayes/reward arms))"
  [bandit]
  (rest (iterate (partial simulate (:bandit bandit) bayes/select-arm bayes/reward)
                 bandit)))

(def conversion-rates
  {:home-page-variant-a-results-page-variant-a 0.05
   :home-page-variant-a-results-page-variant-b  0.045
   :home-page-variant-a-results-page-variant-c  0.055
   :home-page-variant-a-results-page-variant-d  0.048
   :home-page-variant-a-results-page-variant-e  0.06
   :home-page-variant-b-results-page-variant-a  0.03
   :home-page-variant-b-results-page-variant-b  0.04
   :home-page-variant-b-results-page-variant-c  0.043
   :home-page-variant-b-results-page-variant-d  0.053
   :home-page-variant-b-results-page-variant-e  0.043
   :home-page-variant-c-results-page-variant-a  0.042
   :home-page-variant-c-results-page-variant-b  0.041
   :home-page-variant-c-results-page-variant-c  0.05
   :home-page-variant-c-results-page-variant-d  0.038
   :home-page-variant-c-results-page-variant-e  0.044
   :home-page-variant-d-results-page-variant-a  0.054
   :home-page-variant-d-results-page-variant-b  0.045
   :home-page-variant-d-results-page-variant-c  0.043
   :home-page-variant-d-results-page-variant-d  0.039
   :home-page-variant-d-results-page-variant-e  0.042})

(def variants
  [{:name :home-page-variant-a :layer 0}
   {:name :home-page-variant-b :layer 0}
   {:name :home-page-variant-c :layer 0}
   {:name :home-page-variant-d :layer 0}
   {:name :results-page-variant-a :layer 1}
   {:name :results-page-variant-b :layer 1}
   {:name :results-page-variant-c :layer 1}
   {:name :results-page-variant-d :layer 1}
   {:name :results-page-variant-e :layer 1}])

(defn join-names
  [a b]
  (let [a (name (or (:name a) a))
        b (name (or (:name b) b))]
    (keyword 
     (str
      a
      "-"
      b))))

(defn chained-bernoulli-arm
  "creates a function representing the bandit arm. uses a fixed
   probability p of reward. p of 0.1 would reward ~10% of the time."
  [arm-k]
  (fn [second-arm-key]
    (if (> (rand) (conversion-rates (join-names arm-k second-arm-key)))
      0
      1)))


(def initial-chained-state
  (let [layers (map last (group-by :layer variants))]
    (println "Initial chained state initiiaaed")
    {:bandit (doall (for [layer layers]
                      (reduce merge (map (comp (partial apply hash-map)
                                               (juxt :name (comp chained-bernoulli-arm :name)))
                                         layer))))
     :arms   (for [layer layers]
               (apply exp3/bandit (map :name layer)))
     :result {:pulled nil
              :reward 0
              :t 0
              :cumulative-reward 0}}))

(def initial-combinatorial-state
  {:bandit (apply merge
                  (for [first-layer-variant  (filter #(= 0 (:layer %)) variants)
                        second-layer-variant (filter #(= 1 (:layer %)) variants)]
                    (let [joined-name (join-names first-layer-variant
                                                  second-layer-variant)]
                      {joined-name
                       (bernoulli-arm (conversion-rates joined-name))})))
   :arms   (apply exp3/bandit (for [first-layer-variant  (filter #(= 0 (:layer %)) variants)
                                    second-layer-variant (filter #(= 1 (:layer %)) variants)]
                                (join-names first-layer-variant
                                            second-layer-variant)))
   :result {:pulled nil
            :reward 0
            :t 0
            :cumulative-reward 0}})

(defn run-chained-simulation
  []
  (iterate simulate-chained-bandit initial-chained-state))

(def arm-paths
  (map (juxt :layer :name) variants))

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
  (time
   (doall [(for [i (range 100)]
              (do (println "Run " i  " complete")
                  (->> (run-chained-simulation)
                       (take number-of-samples)
                       (map :result)
                       (sort-by :t)
                       last
                       :cumulative-reward
                       (conj []))))])))


(defn calculate-conversions-under-combinatorial-methodology
  [number-of-samples]
  (time
   (doall (for [i (range 100)]
            (do (println "Run " i " complete")
                (->> (simulation-seq-simple initial-combinatorial-state)
                     (take number-of-samples)
                     (map :result)
                     (sort-by :t)
                     last
                     :cumulative-reward
                     (conj [])))))))

(defn write-to-csv
  [data & [filename]]
  (with-open [out-csv (writer (or filename "/tmp/output.csv"))]
    (write-csv out-csv data)))

(defn compare-bandits
  []
  (println "Running combinatorial bandit")
  (write-to-csv (calculate-conversions-under-combinatorial-methodology 100000) "/tmp/combinatorial.csv")
  (println "Running chained bandit")
  (write-to-csv (calculate-conversions-under-combinatorial-methodology 100000) "/tmp/chained.csv"))

(defn -main
  [& args]
  (compare-bandits))

(def current-state
  (atom initial-chained-state))

(defn update-state!
  []
  (swap!
   current-state simulate-chained-bandit))


(def width 1000)
(def height 1000)

(def initial-position
  (let [number-in-each-layer (apply merge
                                    (map (fn [[layer values]]
                                           {layer (count values)})
                                         (group-by :layer variants)))]
    (apply merge
           (map-indexed (fn [i variant]
                          {(:name variant)
                           [(int (+ (/ width 10.0)
                                    (* (:layer variant)
                                       (/ width 5.0))))
                            (int (+ (/ height 10.0)
                                    (* (/ height 10)
                                       (- i (reduce +
                                                    (map #(get number-in-each-layer %) (range (:layer variant))))))))]})
                        (sort-by (comp :name :layer) variants)))))

(defn to-proportion
  [pulls total-pulls]
  (int (* (/ height 5.0) (/ pulls total-pulls))))

(defn draw
  [state]
  (update-state!)
  (q/background 120)
  (let [visitors (-> @current-state :result :t)
        switches (-> @current-state :result :cumulative-reward)
        conversion (/ (int (* (/ switches visitors) 100)) 100.0)]
    (q/text  (str "Number of visitors : " visitors) (* width 0.7) (* height 0.2))
    (q/text  (str "Number of switches : " switches) (* width 0.7) (* height 0.3))
    (q/text  (str "Overall conversion rate : " (* 100.0 conversion) "%") (* width 0.7) (* height 0.4)))
  (let [total-pulls (reduce (fn [agg path] (+ agg (:pulls (get-in (:arms @current-state) path)))) 0 arm-paths)]
    (doseq [arm-path arm-paths]
      (let [arm (get-in (:arms @current-state) arm-path)
            [x y] (initial-position (:name arm))
            proportion (to-proportion (:pulls arm) total-pulls)]
        (q/ellipse x y proportion proportion)
        (q/text (str "Times pulled: "(:pulls arm)) (- x (/ width 20.0)) (+ y (/ height 25)))))))

(defn setup
  []
  (q/frame-rate 50))

(defn draw-it
  []
  (q/defsketch bandit-sketch                  
    :title    "Chained Bandit"    
    :settings #(q/smooth 2)             
    :draw     draw
    :setup    setup
    :features [:keep-on-top]
    :middleware [m/fun-mode m/pause-on-error]
    :size     [width height]))

(defn clear-state
  []
  (reset! @current-state initial-chained-state))

(defn reload
  []
  (use 'bandit.simulate :reload))
