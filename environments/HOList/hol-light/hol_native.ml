(* ========================================================================= *)
(*                               HOL LIGHT                                   *)
(*                                                                           *)
(*              Modern OCaml version of the HOL theorem prover               *)
(*                                                                           *)
(*                            John Harrison                                  *)
(*                                                                           *)
(*            (c) Copyright, University of Cambridge 1998                    *)
(*              (c) Copyright, John Harrison 1998-2007                       *)
(* ========================================================================= *)

set_jrh_lexer;;

let hol_version = "2.20++";;

let hol_dir = ref
  (try Sys.getenv "HOLLIGHT_DIR" with Not_found -> Sys.getcwd());;

(* ------------------------------------------------------------------------- *)
(* Should eventually change to "ref(Filename.temp_dir_name)".                *)
(* However that's not available in 3.08, which is still the default          *)
(* in Cygwin, and I don't want to force people to upgrade Ocaml.             *)
(* ------------------------------------------------------------------------- *)

let temp_path = ref "/tmp";;

(* ------------------------------------------------------------------------- *)
(* Load files from system and/or user-settable directories.                  *)
(* Paths map initial "$/" to !hol_dir dynamically; use $$ to get the actual  *)
(* $ character at the start of a directory.                                  *)
(* ------------------------------------------------------------------------- *)

let hol_expand_directory s =
  if s = "$" || s = "$/" then !hol_dir
  else if s = "$$" then "$"
  else if String.length s <= 2 then s
  else if String.sub s 0 2 = "$$" then (String.sub s 1 (String.length s - 1))
  else if String.sub s 0 2 = "$/"
  then Filename.concat (!hol_dir) (String.sub s 2 (String.length s - 2))
  else s;;

let load_path = ref ["."; "$"];;

let file_on_path p s =
  if not (Filename.is_relative s) then s else
  let p' = List.map hol_expand_directory p in
  let d = List.find (fun d -> Sys.file_exists(Filename.concat d s)) p' in
  Filename.concat (if d = "." then Sys.getcwd() else d) s;;
