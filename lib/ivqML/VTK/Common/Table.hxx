// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================
#ifndef __ivqML__VTK___Common__Table__hxx__
#define __ivqML__VTK___Common__Table__hxx__

// -------------------------------------------------------------------------
template< class _TM >
void ivqML::VTK::Common::Table< _TM >::
Init( _TM& D )
{
  this->_Arrays.clear( );
  this->_Table = vtkSmartPointer< vtkTable >::New( );

  this->_Buffer = D.data( );
  float* b = this->_Buffer;
  for( unsigned long long c = 0; c < D.cols( ); ++c )
  {
    std::stringstream name;
    name << "dim_" << c;

    auto a = vtkSmartPointer< vtkFloatArray >::New( );
    a->SetName( name.str( ).c_str( ) );
    a->SetVoidArray( b, D.rows( ), 1 );
    b += D.rows( );

    this->_Table->AddColumn( a );
    this->_Arrays.push_back( a );
  } // end for
  this->_Table->SetNumberOfRows( D.rows( ) );
  this->_Arrays.shrink_to_fit( );
}

// -------------------------------------------------------------------------
template< class _TM >
void ivqML::VTK::Common::Table< _TM >::
Modified( )
{
  for( auto a: this->_Arrays )
    a->Modified( );
  this->_Table->Modified( );
}

#endif // __ivqML__VTK___Common__Table__hxx__

// eof - $RCSfile$
