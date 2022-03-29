type token = int

module Vector : sig
  type t = float array

  val dot : t -> t -> float
end

type logits = Vector.t

module State : sig
  type t = float array

  val pp : t Fmt.t
  val zero : unit -> t
  val update : t -> t -> t
  val query : t -> t -> float
end

type query = State.t
type update = State.t

module Embedding : sig
  type t = State.t array

  val apply : t -> token -> State.t
end

val softmax : Vector.t -> unit

type attn_vector = Vector.t

module Attn_head : sig
  type t =
    { w_q : State.t -> attn_vector
    ; w_k : State.t -> attn_vector
    ; w_v : State.t -> attn_vector
    ; w_o : State.t -> attn_vector
    }

  val apply : t -> State.t array -> attn_vector array
end

module Attn_layer : sig
  type t = Attn_head.t array

  val apply : t -> State.t array -> update array
end

module Neuron : sig
  type t =
    { read : query
    ; write : update
    }
end

module Mlp_layer : sig
  type t =
    { mlps : Neuron.t array
    ; nonlinear : float -> float
    }

  val apply : t -> State.t -> update
end

module Res_block : sig
  type t =
    { attn : Attn_layer.t
    ; mlps : Mlp_layer.t
    }
end

module Logit_fn : sig
  type t = query

  val apply : t -> State.t -> float
end

module Unembedding : sig
  type t = Logit_fn.t array

  val apply : t -> State.t -> logits
end

module Transformer : sig
  type t =
    { embedding : Embedding.t
    ; layers : Res_block.t array
    ; unembedding : Unembedding.t
    }

  val apply : t -> token array -> logits array
end
