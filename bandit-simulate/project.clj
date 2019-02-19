(defproject bandit/bandit-simulate "0.2.1-SNAPSHOT"
  :description "Multi-armed bandit simulation"
  :url "http://github.com/pingles/bandit"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[bandit/bandit-core "0.2.1-SNAPSHOT"]
                 [org.clojure/clojure "1.6.0"]
                 [org.clojure/math.numeric-tower "0.0.4"]
                 [org.clojure/tools.cli "0.2.2"]
                 [org.clojure/data.csv "0.1.2"]
                 [incanter/incanter-core "1.5.1"]
                 [quil "2.3.0"]]
  :profiles {:dev {:dependencies [[criterium "0.4.1"]
                                  [expectations "1.4.48"]]
                   :plugins [[lein-expectations "0.0.8"]]}}
  :main bandit.simulate
  :min-lein-version "2.0.0"
  :jvm-opts ["-Xmx2G" "-server" "-XX:+UseConcMarkSweepGC"])
