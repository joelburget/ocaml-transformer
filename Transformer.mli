type token = int

(** A basic 1-dimensional vector. *)
module Vector : sig
  type t = float array

  val dot : t -> t -> float
end

(** Logits represent the model's prediction for the next token (one logit value per entry
    in the vocabulary). *)
type logits = Vector.t

(** Representation of data seen after every token. ("hidden state") *)
module State : sig
  type t = float array

  val pp : t Fmt.t
  val zero : unit -> t
  val update : t -> t -> t
  val query : t -> t -> float
end

(** Vector used to query the state. *)
type query = State.t

(** Vector used to update the state (by addition). *)
type update = State.t

(** Embed a token (to a state). *)
module Embedding : sig
  type t = State.t array

  val apply : t -> token -> State.t
end

val softmax : Vector.t -> unit

type attn_vector = Vector.t

(** Attention heads transform the residual stream via query-key and output-value circuits. *)
module Attn_head : sig
  type t =
    { w_q : State.t -> attn_vector (** query *)
    ; w_k : State.t -> attn_vector (** key *)
    ; w_v : State.t -> attn_vector (** value *)
    ; w_o : State.t -> update (** output *)
    }

  val apply : t -> State.t array -> attn_vector array
end

(** An attention layer is multiple ([d_head = 128]) heads operating in parallel. *)
module Attn_layer : sig
  type t = Attn_head.t array

  val apply : t -> State.t array -> update array
end

(** A neuron reads state, runs an activation function, and outputs state. *)
module Neuron : sig
  type t =
    { read : query
    ; write : update
    }

  val apply : t -> (float -> float) -> State.t -> update
end

(** An mlp layer is a set of neurons each reading from the same input state and
    contributing to the same output. *)
module Mlp_layer : sig
  type t =
    { mlps : Neuron.t array
    ; nonlinear : float -> float
    }

  val apply : t -> State.t -> update
end

(** A residual block is an attention layer followed by an MLP layer. *)
module Res_block : sig
  type t =
    { attn : Attn_layer.t
    ; mlps : Mlp_layer.t
    }
end

(** Logits query a state, generating a prediction. *)
module Logit_fn : sig
  type t = query

  val apply : t -> State.t -> float
end

(** Unembed the state to produce an array of predictions (logits). *)
module Unembedding : sig
  type t = Logit_fn.t array

  val apply : t -> State.t -> logits
end

(* A transformer embeds tokens, passes them through the residual block layers, then unembeds them to a prediction. *)
module Transformer : sig
  type t =
    { embedding : Embedding.t
    ; layers : Res_block.t array
    ; unembedding : Unembedding.t
    }

  val apply : t -> token array -> logits array
end
