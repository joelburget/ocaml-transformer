open Base
(* TODO: Ocaml doesn't really have 32-bit (or smaller) floats *)

let n_layers = 96
let d_model = 128 * n_layers

(* let d_mlp = 4 * d_model *)
let d_head = 128
(* let n_heads = d_model / d_head *)
(* let n_vocab = 50_000 *)

type token = int (* TODO: unsigned *)

let pp_array = Fmt.(brackets (array ~sep:semi float))

module Vector = struct
  type t = float array

  let dot l r =
    Array.zip_exn l r |> Array.fold ~init:0.0 ~f:(fun accum (l, r) -> accum +. (l *. r))
  ;;

  let%test_module _ =
    (module struct
      let go l r = Fmt.pr "%f@." (dot l r)

      let%expect_test _ =
        go [||] [||];
        go [| 1.; 1. |] [| 1.; 1. |];
        go [| 1.; 2. |] [| 2.; 1. |];
        [%expect {|
        0.000000
        2.000000
        4.000000
        |}]
      ;;
    end)
  ;;
end

type logits = Vector.t

module State = struct
  type t = float array

  let pp = pp_array
  let zero () = Array.create ~len:d_model 0.0

  let update self update =
    let out = Array.copy self in
    Array.iteri update ~f:(fun i r -> out.(i) <- out.(i) +. r);
    out
  ;;

  let query self right = Vector.dot self right

  let%test_module _ =
    (module struct
      let%expect_test _ =
        Fmt.pr "%a@." pp [| 1.; 2.; 3. |];
        [%expect {| [1; 2; 3] |}]
      ;;

      let%expect_test _ =
        Fmt.pr "%a@." pp (update [| 1.; 2.; 3. |] [| 1.; 2.; 3. |]);
        [%expect {| [2; 4; 6] |}]
      ;;

      let%expect_test _ =
        Fmt.pr "%f@." (query [| 1.; 2.; 3. |] [| 3.; 2.; 1. |]);
        [%expect {| 10.000000 |}]
      ;;
    end)
  ;;
end

type query = State.t
type update = State.t

module Embedding = struct
  type t = State.t array

  let apply self tok : State.t = Array.copy self.(tok)
end

type attn_vector = float array

let softmax (scores : float array) =
  let exp_sum = ref 0.0 in
  for i = 0 to Array.length scores - 1 do
    let e_x = Float.exp scores.(i) in
    scores.(i) <- e_x;
    exp_sum := !exp_sum +. e_x
  done;
  for i = 0 to Array.length scores - 1 do
    scores.(i) <- scores.(i) /. !exp_sum
  done
;;

let%expect_test _ =
  let go arr =
    softmax arr;
    Fmt.pr "%a@." pp_array arr
  in
  go [| 0. |];
  go [| 1. |];
  go [| 1.; 2.; 3.; 4.; 1.; 2.; 3. |];
  [%expect
    {|
    [1]
    [1]
    [0.0236405; 0.0642617; 0.174681; 0.474833; 0.0236405; 0.0642617; 0.174681] |}]
;;

module Attn_head = struct
  type t =
    { w_q : State.t -> attn_vector
    ; w_k : State.t -> attn_vector
    ; w_v : State.t -> attn_vector
    ; w_o : State.t -> attn_vector
    }

  let apply self states =
    let qs : attn_vector array = Array.map ~f:self.w_q states in
    let ks : attn_vector array = Array.map ~f:self.w_k states in
    let vs : attn_vector array = Array.map ~f:self.w_v states in
    let values = Array.map states ~f:(fun _ -> Array.create ~len:d_head 0.0) in
    Array.iteri qs ~f:(fun src my_q ->
        let scores : float array = Array.init src ~f:(fun i -> Vector.dot my_q ks.(i)) in
        softmax scores;
        Array.zip_exn scores vs
        |> Array.iter ~f:(fun (score, v) ->
               Array.iteri v ~f:(fun i v ->
                   values.(src).(i) <- values.(src).(i) +. (v *. score))));
    Array.map values ~f:self.w_o
  ;;
end

module Attn_layer = struct
  type t = Attn_head.t array

  let apply self states : update array =
    let updates : update array ref = ref (Array.map states ~f:(fun _ -> State.zero ())) in
    Array.iter self ~f:(fun h ->
        let head_out = Attn_head.apply h states in
        updates
          := Array.zip_exn !updates head_out
             |> Array.map ~f:(fun (l, r) -> State.update l r));
    !updates
  ;;
end

module Neuron = struct
  type t =
    { read : query
    ; write : update
    }
end

module Mlp_layer = struct
  type t =
    { mlps : Neuron.t array
    ; nonlinear : float -> float
    }

  let apply self state : update =
    let out = ref (State.zero ()) in
    Array.iter self.mlps ~f:(fun mlp ->
        let pre_act = State.query mlp.Neuron.read state in
        let post_act = self.nonlinear pre_act in
        let unit_out : update = Array.map mlp.Neuron.write ~f:(fun f -> f *. post_act) in
        out := State.update !out unit_out);
    !out
  ;;
end

module Res_block = struct
  type t =
    { attn : Attn_layer.t
    ; mlps : Mlp_layer.t
    }
end

module Logit_fn = struct
  type t = query

  let apply self state : float = State.query self state
end

module Unembedding = struct
  type t = Logit_fn.t array

  let apply self state : logits = Array.map self ~f:(fun f -> Logit_fn.apply f state)
end

module Transformer = struct
  type t =
    { embedding : Embedding.t
    ; layers : Res_block.t array
    ; unembedding : Unembedding.t
    }

  let apply self tokens : logits array =
    let states = tokens |> Array.map ~f:(Embedding.apply self.embedding) |> ref in
    Array.iter self.layers ~f:(fun layer ->
        let attn_out = Attn_layer.apply layer.Res_block.attn !states in
        states
          := Array.zip_exn !states attn_out
             |> Array.map ~f:(fun (l, r) -> State.update l r);
        let states = !states in
        for i = 0 to Array.length states - 1 do
          let mlp_out = Mlp_layer.apply layer.Res_block.mlps states.(i) in
          states.(i) <- State.update states.(i) mlp_out
        done);
    Array.map !states ~f:(Unembedding.apply self.unembedding)
  ;;
end
