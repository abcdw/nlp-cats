(ns nlp-cats.core
  (:require [opennlp.tools.train :as train]
            [opennlp.nlp :as nlp]
            [clojure.string :as str]))


(defn train-and-test-model [train-dataset test-dataset]
  (let [cat-model (train/train-document-categorization "fr" train-dataset)
        categorizer (nlp/make-document-categorizer cat-model)
        test-data (str/split (slurp test-dataset) #"\n")]

    (for [line test-data
          :let [[category content] (str/split line #" " 2)
                best-cat (:best-category (categorizer content))]]
      [(= category best-cat)
       category
       content
       (->> (categorizer content)
            meta
            :probabilities
            (into [])
            (sort #(> (last %1) (last %2))))])))

;; (def nonlazy-results (into [] results))
;; (spit "test.file" nonlazy-results)

;; (def results (get-results "train.csv" "test.csv"))

;; (clojure.pprint/pprint results)
(defn count-matches [results]
  [(reduce #(+ (if (first %2) 1 0) %1) 0 results) (count results)])

(defn count-semi-matches [results]
  [(reduce (fn [a b]
             (let [elem (last b)
                   real-cat (second b)
                   two-cats #{(first (first elem))
                              (first (second elem))}]
               (+ (if (two-cats real-cat) 1 0) a)))
           0 results)
   (count results)])

(defn remove-indexed [v n]
  (into (subvec v 0 n) (subvec v (inc n))))

(defn do-cross-validation [matcher]
  (let [dataset (-> (slurp "data.csv")
                    (str/split #"\n"))]
    (reduce #(map + %1 %2) [0 0]
            (for [x (range (count dataset))
                  :let [train-data (remove-indexed dataset x)
                        test-data [(nth dataset x)]]]
              (do (spit "train.ds" (str/join "\n" train-data))
                  (spit "test.ds" (str/join "\n" test-data))
                  (matcher (train-and-test-model "train.ds" "test.ds")))))))

;; (apply map + [[1 2] [2 5]])

(do-cross-validation count-semi-matches)
;; (/ 27 45)
;; (/ 33 45)
;; (count-matches results)


(clojure.pprint/pprint results)
