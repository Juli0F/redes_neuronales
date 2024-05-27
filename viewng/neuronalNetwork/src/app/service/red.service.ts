import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { retry } from 'rxjs';

@Injectable({
  providedIn: 'root'
})
export class RedService {

  private apiUrl = "http://localhost:8000" ;

  constructor(private http: HttpClient) { }
  
  train(createNeuronal:any){
    return this.http.post<string>(this.apiUrl, createNeuronal);
  }
  testPredict(test:number[][]){
    return this.http.put<number[][]>(this.apiUrl, test);
  }
}
