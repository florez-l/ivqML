// =========================================================================
// @author Leonardo Florez-Valencia (florez-l@javeriana.edu.co)
// =========================================================================

#include <algorithm>
#include <iostream>
#include <ivqML/IO.h>
#include <vector>


namespace ivqML
{
  namespace Model
  {
    namespace NeuralNetwork
    {
      /**
       */
      template< class _TReal, class _TNatural = unsigned long long >
      class FeedForward
      {
      public:
        using TReal    = _TReal;
        using TNatural = _TNatural;
        using TMatrix  = Eigen::Matrix< TReal, Eigen::Dynamic, Eigen::Dynamic >;
        using TColumn  = Eigen::Matrix< TReal, Eigen::Dynamic, 1 >;
        using TRow     = Eigen::Matrix< TReal, 1, Eigen::Dynamic >;

        using TMatrixMap = Eigen::Map< TMatrix >;
        using TColumnMap = Eigen::Map< TColumn >;
        using TRowMap    = Eigen::Map< TRow >;

      public:
        FeedForward( )
          {
          }
        virtual ~FeedForward( )
          {
            this->_free_buffer( );
          }

        void set_input_size( const TNatural& n0 )
          {
            this->m_N.clear( );
            this->m_W.clear( );
            this->m_B.clear( );

            this->m_N.push_back( n0 );
          }
        void add_layer( const TNatural& n, const std::string& a )
          {
            this->m_N.push_back( n );
          }
        void init( )
          {
            this->m_N.shrink_to_fit( );
            this->m_W.clear( );
            this->m_B.clear( );

            this->m_S = 0;
            this->m_BufferSize = 0;
            for( TNatural n = 1; n < this->m_N.size( ); n++ )
            {
              this->m_S += ( this->m_N[ n - 1 ] + 1 ) * this->m_N[ n ];
              this->m_BufferSize += this->m_N[ n ];
            } // end for

            this->m_P = reinterpret_cast< TReal* >( std::calloc( this->m_S, sizeof( TReal ) ) );
            std::fill( this->m_P, this->m_P + this->m_S, TReal( 0 ) );
            TReal* p = this->m_P;
            for( TNatural l = 1; l < this->m_N.size( ); ++l )
            {
              this->m_W.push_back( TMatrixMap( p, this->m_N[ l - 1 ], this->m_N[ l ] ) );
              p += this->m_W.back( ).size( );
              this->m_B.push_back( TColumnMap( p, this->m_N[ l ], 1 ) );
              p += this->m_B.back( ).size( );
            } // end for
          }

        template< class _TX >
        auto operator()( const Eigen::EigenBase< _TX >& bX ) const
          {
            auto X = bX.derived( ).template cast< TReal >( );
            TNatural M = X.cols( );
            if( this->m_BufferSamples < M )
              this->_allocate_buffer( M );

            TReal* bZ = this->m_BufferZ;
            TReal* bA = this->m_BufferA;

            TMatrixMap Z( bZ, this->m_N[ 1 ], M );
            TMatrixMap A( bA, this->m_N[ 1 ], M );

            Z = ( this->m_W[ 0 ] * X ).colwise( ) + this->m_B[ 0 ];
            A = Z;
            for( TNatural l = 1; l < this->m_W.size( ); ++l )
            {
              bZ += Z.size( );
              new ( &Z ) TMatrixMap( bZ, this->m_N[ l ], M );

              bA += A.size( );
            } // end for


            /* TODO
               Z[ 0 ] = ( this->W[ 0 ] * X.derived( ).template cast< TReal >( ) ).colwise( ) + this->B[ 0 ];
               A[ 0 ] = Z[ 0 ];
               for( unsigned long long l = 1; l < this->B.size( ); ++l )
               {
               Z[ l ] = ( this->W[ l ] * A[ l - 1 ] ).colwise( ) + this->B[ l ];
               A[ l ] = Z[ l ];
               } // end if

               return( A.back( ) );
            */
            return( X.derived( ) );
          }

      protected:
        void _allocate_buffer( const TNatural& M ) const
          {
            if( this->m_BufferSamples != M )
            {
              this->_free_buffer( );
              this->m_BufferSamples = M;
              this->m_Buffer = reinterpret_cast< TReal* >( std::calloc( ( this->m_BufferSize * this->m_BufferSamples ) << 1, sizeof( TReal ) ) );
              this->m_BufferZ = this->m_Buffer;
              this->m_BufferA = this->m_Buffer + ( this->m_BufferSize * this->m_BufferSamples );
            } // end if
          }
        void _free_buffer( ) const
          {
            this->m_BufferSamples = 0;
            if( this->m_Buffer != nullptr )
              std::free( this->m_Buffer );
            this->m_Buffer = nullptr;
            this->m_BufferZ = nullptr;
            this->m_BufferA = nullptr;
          }

      protected:
        TNatural m_S { 0 };
        TReal* m_P { nullptr };

        mutable TNatural m_BufferSize    { 0 };
        mutable TNatural m_BufferSamples { 0 };
        mutable TReal*   m_Buffer { nullptr };
        mutable TReal*   m_BufferZ { nullptr };
        mutable TReal*   m_BufferA { nullptr };

        std::vector< TNatural > m_N;
        std::vector< TMatrixMap > m_W;
        std::vector< TColumnMap > m_B;
      };
    } // end namespace
  } // end namespace
} // end namespace

/* TODO
   class Layer
   {
   public:
   using TReal = long double;
   using TMatrix = Eigen::Matrix< TReal, Eigen::Dynamic, Eigen::Dynamic >;
   using TColumn = Eigen::Matrix< TReal, Eigen::Dynamic, 1 >;

   Layer( )
   {
   this->W.push_back( TMatrix::Zero( 40, 784 ) );
   this->W.push_back( TMatrix::Zero( 20, 40 ) );
   this->W.push_back( TMatrix::Zero( 10, 20 ) );

   this->B.push_back( TColumn::Zero( 40 ) );
   this->B.push_back( TColumn::Zero( 20 ) );
   this->B.push_back( TColumn::Zero( 10 ) );

   this->N = 40 + 20 + 10;

   std::cout << this->N << std::endl;
   }

   virtual ~Layer( )
   {
   if( this->D != nullptr )
   std::free( this->D );
   }

   template< class _TX >
   auto operator()( const Eigen::EigenBase< _TX >& X ) const
   {
   std::vector< Eigen::Map< TMatrix > > Z, A;
   if( this->M < X.cols( ) )
   {
        
   std::cout << "------------------------" << std::endl;
   std::cout << "allocate!!!" << std::endl;
   std::cout << "------------------------" << std::endl;

   this->M = X.cols( );
   if( this->D != nullptr )
   std::free( this->D );
   this->D = reinterpret_cast< TReal* >( std::calloc( ( this->N * this->M ) << 1, sizeof( TReal ) ) );
   } // end if

   TReal* wD = this->D;
   TReal* aD = this->D + ( this->N * this->M );
   for( unsigned long long l = 0; l < this->B.size( ); ++l )
   {
   Z.push_back( Eigen::Map< TMatrix >( wD, this->B[ l ].size( ), X.cols( ) ) );
   A.push_back( Eigen::Map< TMatrix >( aD, this->B[ l ].size( ), X.cols( ) ) );
   wD += Z.back( ).size( );
   aD += Z.back( ).size( );
   } // end for

   Z[ 0 ] = ( this->W[ 0 ] * X.derived( ).template cast< TReal >( ) ).colwise( ) + this->B[ 0 ];
   A[ 0 ] = Z[ 0 ];
   for( unsigned long long l = 1; l < this->B.size( ); ++l )
   {
   Z[ l ] = ( this->W[ l ] * A[ l - 1 ] ).colwise( ) + this->B[ l ];
   A[ l ] = Z[ l ];
   } // end if

   return( A.back( ) );
   }

   protected:
   mutable std::vector< TMatrix > W;
   mutable std::vector< TColumn > B;

   mutable unsigned long long N { 0 };
   mutable unsigned long long M { 0 };
   mutable TReal* D { nullptr };
   };
*/

int main( int argc, char** argv )
{
  using TReal = long double;
  using TModel = ivqML::Model::NeuralNetwork::FeedForward< TReal >;

  TModel::TMatrix Xtr, Ytr, Xte, Yte;
  ivqML::IO::ReadMNIST( Xtr, Ytr, Xte, Yte, argv[ 1 ] );

  /* TODO
     TMatrix W = TMatrix::Zero( 40, Xtr.rows( ) );
     TColumn B = TColumn::Zero( 40 );
  */

  TModel model;
  model.set_input_size( Xtr.rows( ) );
  model.add_layer( 40, "relu" );
  model.add_layer( 20, "ReLu" );
  model.add_layer( 10, "softMax" );
  model.init( );

  TModel::TMatrix Atr = model( Xtr );
  TModel::TMatrix Ate = model( Xte );

  /* TODO
     Layer model;
     auto Y = model( Xtr );
     std::cout << "_Z" << typeid( Y ).name( ) << std::endl;
     std::cout << Xtr.rows( ) << " " << Xtr.cols( ) << std::endl;
     std::cout << Y.rows( ) << " " << Y.cols( ) << std::endl;
     auto Yo = model( Xte );
     std::cout << "_Z" << typeid( Yo ).name( ) << std::endl;
     std::cout << Xte.rows( ) << " " << Xte.cols( ) << std::endl;
     std::cout << Yo.rows( ) << " " << Yo.cols( ) << std::endl;
  */

  /* TODO
     auto Y = ( Eigen::Map< TMatrix >( W.data( ), W.rows( ), W.cols( ) ) * Xtr ).colwise( ) + Eigen::Map< TColumn >( B.data( ), B.rows( ), B.cols( ) );

     std::cout << Xtr.rows( ) << " " << Xtr.cols( ) << std::endl;
     std::cout << Y.rows( ) << " " << Y.cols( ) << std::endl;
     std::cout << "_Z" << typeid( Y ).name( ) << std::endl;
     Eigen::CwiseBinaryOp<Eigen::internal::scalar_sum_op<TReal, TReal>, const Eigen::Product<TMatrixMap, TMatrix, 0>, const Eigen::Replicate<TColumnMap, 1, -1> >
  */

  /* TODO
     std::cout << Xtr.rows( ) << " " << Xtr.cols( ) << std::endl;
     std::cout << Ytr.rows( ) << " " << Ytr.cols( ) << std::endl;
     std::cout << Xte.rows( ) << " " << Xte.cols( ) << std::endl;
     std::cout << Yte.rows( ) << " " << Yte.cols( ) << std::endl;
  */

  /* TODO
     std::cout << "P2" << std::endl;
     std::cout << "28 28" << std::endl;
     std::cout << "255" << std::endl;
     for( unsigned long long c = 0; c < Xtr.cols( ); ++c )
     std::cout << int( Xtr( 55554, c ) ) << " ";
     std::cout << std::endl;
  */


  return( EXIT_SUCCESS );
}

// eof - $RCSfile$
