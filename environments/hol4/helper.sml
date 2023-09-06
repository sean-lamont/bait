load "Timeout";

val default_pt = get_term_printer();

(* fun pt t = *)
(*     case dest_term t of *)
(*         VAR (nm, _) => "V" ^ nm *)
(*       | CONST {Name,Thy,...} => "C" ^ Thy ^ "$" ^ Name *)
(*       | COMB(t1,t2) => "@ " ^ pt t1 ^ " " ^ pt t2 *)
(*       | LAMB(v,b) => "| " ^ pt v ^ " " ^ pt b; *)

fun pt t =
    case dest_term t of
        VAR (nm, _) => "V" ^ nm
      | CONST {Name,Thy,...} => "C" ^ "$" ^ Thy ^ "$ " ^ Name
      | COMB(t1,t2) => "@ " ^ pt t1 ^ " " ^ pt t2
      | LAMB(v,b) => "| " ^ pt v ^ " " ^ pt b;

(* fun my_top_goals gl = term_to_string gl *)

(* val _ = set_term_printer (HOLPP.add_string o pt) *)

(* val _ = set_term_printer default_pt *)

fun e tac = Timeout.apply (Time.fromReal 0.1) proofManagerLib.e tac

fun unlimited_e tac = proofManagerLib.e tac
